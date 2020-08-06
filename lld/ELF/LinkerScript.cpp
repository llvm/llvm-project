//===- LinkerScript.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the parser/evaluator of the linker script.
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::elf;

LinkerScript *elf::script;

static uint64_t getOutputSectionVA(SectionBase *sec) {
  OutputSection *os = sec->getOutputSection();
  assert(os && "input section has no output section assigned");
  return os ? os->addr : 0;
}

uint64_t ExprValue::getValue() const {
  if (sec)
    return alignTo(sec->getOffset(val) + getOutputSectionVA(sec),
                   alignment);
  return alignTo(val, alignment);
}

uint64_t ExprValue::getSecAddr() const {
  if (sec)
    return sec->getOffset(0) + getOutputSectionVA(sec);
  return 0;
}

uint64_t ExprValue::getSectionOffset() const {
  // If the alignment is trivial, we don't have to compute the full
  // value to know the offset. This allows this function to succeed in
  // cases where the output section is not yet known.
  if (alignment == 1 && !sec)
    return val;
  return getValue() - getSecAddr();
}

OutputSection *LinkerScript::createOutputSection(StringRef name,
                                                 StringRef location) {
  OutputSection *&secRef = nameToOutputSection[name];
  OutputSection *sec;
  if (secRef && secRef->location.empty()) {
    // There was a forward reference.
    sec = secRef;
  } else {
    sec = make<OutputSection>(name, SHT_PROGBITS, 0);
    if (!secRef)
      secRef = sec;
  }
  sec->location = std::string(location);
  return sec;
}

OutputSection *LinkerScript::getOrCreateOutputSection(StringRef name) {
  OutputSection *&cmdRef = nameToOutputSection[name];
  if (!cmdRef)
    cmdRef = make<OutputSection>(name, SHT_PROGBITS, 0);
  return cmdRef;
}

// Expands the memory region by the specified size.
static void expandMemoryRegion(MemoryRegion *memRegion, uint64_t size,
                               StringRef regionName, StringRef secName) {
  memRegion->curPos += size;
  uint64_t newSize = memRegion->curPos - (memRegion->origin)().getValue();
  uint64_t length = (memRegion->length)().getValue();
  if (newSize > length)
    error("section '" + secName + "' will not fit in region '" + regionName +
          "': overflowed by " + Twine(newSize - length) + " bytes");
}

void LinkerScript::expandMemoryRegions(uint64_t size) {
  if (ctx->memRegion)
    expandMemoryRegion(ctx->memRegion, size, ctx->memRegion->name,
                       ctx->outSec->name);
  // Only expand the LMARegion if it is different from memRegion.
  if (ctx->lmaRegion && ctx->memRegion != ctx->lmaRegion)
    expandMemoryRegion(ctx->lmaRegion, size, ctx->lmaRegion->name,
                       ctx->outSec->name);
}

void LinkerScript::expandOutputSection(uint64_t size) {
  ctx->outSec->size += size;
  expandMemoryRegions(size);
}

void LinkerScript::setDot(Expr e, const Twine &loc, bool inSec) {
  uint64_t val = e().getValue();
  if (val < dot && inSec)
    error(loc + ": unable to move location counter backward for: " +
          ctx->outSec->name);

  // Update to location counter means update to section size.
  if (inSec)
    expandOutputSection(val - dot);

  dot = val;
}

// Used for handling linker symbol assignments, for both finalizing
// their values and doing early declarations. Returns true if symbol
// should be defined from linker script.
static bool shouldDefineSym(SymbolAssignment *cmd) {
  if (cmd->name == ".")
    return false;

  if (!cmd->provide)
    return true;

  // If a symbol was in PROVIDE(), we need to define it only
  // when it is a referenced undefined symbol.
  Symbol *b = symtab->find(cmd->name);
  if (b && !b->isDefined())
    return true;
  return false;
}

// Called by processSymbolAssignments() to assign definitions to
// linker-script-defined symbols.
void LinkerScript::addSymbol(SymbolAssignment *cmd) {
  if (!shouldDefineSym(cmd))
    return;

  // Define a symbol.
  ExprValue value = cmd->expression();
  SectionBase *sec = value.isAbsolute() ? nullptr : value.sec;
  uint8_t visibility = cmd->hidden ? STV_HIDDEN : STV_DEFAULT;

  // When this function is called, section addresses have not been
  // fixed yet. So, we may or may not know the value of the RHS
  // expression.
  //
  // For example, if an expression is `x = 42`, we know x is always 42.
  // However, if an expression is `x = .`, there's no way to know its
  // value at the moment.
  //
  // We want to set symbol values early if we can. This allows us to
  // use symbols as variables in linker scripts. Doing so allows us to
  // write expressions like this: `alignment = 16; . = ALIGN(., alignment)`.
  uint64_t symValue = value.sec ? 0 : value.getValue();

  Defined newSym(nullptr, cmd->name, STB_GLOBAL, visibility, STT_NOTYPE,
                 symValue, 0, sec);

  Symbol *sym = symtab->insert(cmd->name);
  sym->mergeProperties(newSym);
  sym->replace(newSym);
  cmd->sym = cast<Defined>(sym);
}

// This function is called from LinkerScript::declareSymbols.
// It creates a placeholder symbol if needed.
static void declareSymbol(SymbolAssignment *cmd) {
  if (!shouldDefineSym(cmd))
    return;

  uint8_t visibility = cmd->hidden ? STV_HIDDEN : STV_DEFAULT;
  Defined newSym(nullptr, cmd->name, STB_GLOBAL, visibility, STT_NOTYPE, 0, 0,
                 nullptr);

  // We can't calculate final value right now.
  Symbol *sym = symtab->insert(cmd->name);
  sym->mergeProperties(newSym);
  sym->replace(newSym);

  cmd->sym = cast<Defined>(sym);
  cmd->provide = false;
  sym->scriptDefined = true;
}

using SymbolAssignmentMap =
    DenseMap<const Defined *, std::pair<SectionBase *, uint64_t>>;

// Collect section/value pairs of linker-script-defined symbols. This is used to
// check whether symbol values converge.
static SymbolAssignmentMap
getSymbolAssignmentValues(const std::vector<BaseCommand *> &sectionCommands) {
  SymbolAssignmentMap ret;
  for (BaseCommand *base : sectionCommands) {
    if (auto *cmd = dyn_cast<SymbolAssignment>(base)) {
      if (cmd->sym) // sym is nullptr for dot.
        ret.try_emplace(cmd->sym,
                        std::make_pair(cmd->sym->section, cmd->sym->value));
      continue;
    }
    for (BaseCommand *sub_base : cast<OutputSection>(base)->sectionCommands)
      if (auto *cmd = dyn_cast<SymbolAssignment>(sub_base))
        if (cmd->sym)
          ret.try_emplace(cmd->sym,
                          std::make_pair(cmd->sym->section, cmd->sym->value));
  }
  return ret;
}

// Returns the lexicographical smallest (for determinism) Defined whose
// section/value has changed.
static const Defined *
getChangedSymbolAssignment(const SymbolAssignmentMap &oldValues) {
  const Defined *changed = nullptr;
  for (auto &it : oldValues) {
    const Defined *sym = it.first;
    if (std::make_pair(sym->section, sym->value) != it.second &&
        (!changed || sym->getName() < changed->getName()))
      changed = sym;
  }
  return changed;
}

// Process INSERT [AFTER|BEFORE] commands. For each command, we move the
// specified output section to the designated place.
void LinkerScript::processInsertCommands() {
  for (const InsertCommand &cmd : insertCommands) {
    // If cmd.os is empty, it may have been discarded by
    // adjustSectionsBeforeSorting(). We do not handle such output sections.
    auto from = llvm::find(sectionCommands, cmd.os);
    if (from == sectionCommands.end())
      continue;
    sectionCommands.erase(from);

    auto insertPos = llvm::find_if(sectionCommands, [&cmd](BaseCommand *base) {
      auto *to = dyn_cast<OutputSection>(base);
      return to != nullptr && to->name == cmd.where;
    });
    if (insertPos == sectionCommands.end()) {
      error("unable to insert " + cmd.os->name +
            (cmd.isAfter ? " after " : " before ") + cmd.where);
    } else {
      if (cmd.isAfter)
        ++insertPos;
      sectionCommands.insert(insertPos, cmd.os);
    }
  }
}

// Symbols defined in script should not be inlined by LTO. At the same time
// we don't know their final values until late stages of link. Here we scan
// over symbol assignment commands and create placeholder symbols if needed.
void LinkerScript::declareSymbols() {
  assert(!ctx);
  for (BaseCommand *base : sectionCommands) {
    if (auto *cmd = dyn_cast<SymbolAssignment>(base)) {
      declareSymbol(cmd);
      continue;
    }

    // If the output section directive has constraints,
    // we can't say for sure if it is going to be included or not.
    // Skip such sections for now. Improve the checks if we ever
    // need symbols from that sections to be declared early.
    auto *sec = cast<OutputSection>(base);
    if (sec->constraint != ConstraintKind::NoConstraint)
      continue;
    for (BaseCommand *base2 : sec->sectionCommands)
      if (auto *cmd = dyn_cast<SymbolAssignment>(base2))
        declareSymbol(cmd);
  }
}

// This function is called from assignAddresses, while we are
// fixing the output section addresses. This function is supposed
// to set the final value for a given symbol assignment.
void LinkerScript::assignSymbol(SymbolAssignment *cmd, bool inSec) {
  if (cmd->name == ".") {
    setDot(cmd->expression, cmd->location, inSec);
    return;
  }

  if (!cmd->sym)
    return;

  ExprValue v = cmd->expression();
  if (v.isAbsolute()) {
    cmd->sym->section = nullptr;
    cmd->sym->value = v.getValue();
  } else {
    cmd->sym->section = v.sec;
    cmd->sym->value = v.getSectionOffset();
  }
}

static std::string getFilename(InputFile *file) {
  if (!file)
    return "";
  if (file->archiveName.empty())
    return std::string(file->getName());
  return (file->archiveName + ':' + file->getName()).str();
}

bool LinkerScript::shouldKeep(InputSectionBase *s) {
  if (keptSections.empty())
    return false;
  std::string filename = getFilename(s->file);
  for (InputSectionDescription *id : keptSections)
    if (id->filePat.match(filename))
      for (SectionPattern &p : id->sectionPatterns)
        if (p.sectionPat.match(s->name) &&
            (s->flags & id->withFlags) == id->withFlags &&
            (s->flags & id->withoutFlags) == 0)
          return true;
  return false;
}

// A helper function for the SORT() command.
static bool matchConstraints(ArrayRef<InputSectionBase *> sections,
                             ConstraintKind kind) {
  if (kind == ConstraintKind::NoConstraint)
    return true;

  bool isRW = llvm::any_of(
      sections, [](InputSectionBase *sec) { return sec->flags & SHF_WRITE; });

  return (isRW && kind == ConstraintKind::ReadWrite) ||
         (!isRW && kind == ConstraintKind::ReadOnly);
}

static void sortSections(MutableArrayRef<InputSectionBase *> vec,
                         SortSectionPolicy k) {
  auto alignmentComparator = [](InputSectionBase *a, InputSectionBase *b) {
    // ">" is not a mistake. Sections with larger alignments are placed
    // before sections with smaller alignments in order to reduce the
    // amount of padding necessary. This is compatible with GNU.
    return a->alignment > b->alignment;
  };
  auto nameComparator = [](InputSectionBase *a, InputSectionBase *b) {
    return a->name < b->name;
  };
  auto priorityComparator = [](InputSectionBase *a, InputSectionBase *b) {
    return getPriority(a->name) < getPriority(b->name);
  };

  switch (k) {
  case SortSectionPolicy::Default:
  case SortSectionPolicy::None:
    return;
  case SortSectionPolicy::Alignment:
    return llvm::stable_sort(vec, alignmentComparator);
  case SortSectionPolicy::Name:
    return llvm::stable_sort(vec, nameComparator);
  case SortSectionPolicy::Priority:
    return llvm::stable_sort(vec, priorityComparator);
  }
}

// Sort sections as instructed by SORT-family commands and --sort-section
// option. Because SORT-family commands can be nested at most two depth
// (e.g. SORT_BY_NAME(SORT_BY_ALIGNMENT(.text.*))) and because the command
// line option is respected even if a SORT command is given, the exact
// behavior we have here is a bit complicated. Here are the rules.
//
// 1. If two SORT commands are given, --sort-section is ignored.
// 2. If one SORT command is given, and if it is not SORT_NONE,
//    --sort-section is handled as an inner SORT command.
// 3. If one SORT command is given, and if it is SORT_NONE, don't sort.
// 4. If no SORT command is given, sort according to --sort-section.
static void sortInputSections(MutableArrayRef<InputSectionBase *> vec,
                              const SectionPattern &pat) {
  if (pat.sortOuter == SortSectionPolicy::None)
    return;

  if (pat.sortInner == SortSectionPolicy::Default)
    sortSections(vec, config->sortSection);
  else
    sortSections(vec, pat.sortInner);
  sortSections(vec, pat.sortOuter);
}

// Compute and remember which sections the InputSectionDescription matches.
std::vector<InputSectionBase *>
LinkerScript::computeInputSections(const InputSectionDescription *cmd,
                                   ArrayRef<InputSectionBase *> sections) {
  std::vector<InputSectionBase *> ret;

  // Collects all sections that satisfy constraints of Cmd.
  for (const SectionPattern &pat : cmd->sectionPatterns) {
    size_t sizeBefore = ret.size();

    for (InputSectionBase *sec : sections) {
      if (!sec->isLive() || sec->parent)
        continue;

      // For -emit-relocs we have to ignore entries like
      //   .rela.dyn : { *(.rela.data) }
      // which are common because they are in the default bfd script.
      // We do not ignore SHT_REL[A] linker-synthesized sections here because
      // want to support scripts that do custom layout for them.
      if (isa<InputSection>(sec) &&
          cast<InputSection>(sec)->getRelocatedSection())
        continue;

      // Check the name early to improve performance in the common case.
      if (!pat.sectionPat.match(sec->name))
        continue;

      std::string filename = getFilename(sec->file);
      if (!cmd->filePat.match(filename) ||
          pat.excludedFilePat.match(filename) ||
          (sec->flags & cmd->withFlags) != cmd->withFlags ||
          (sec->flags & cmd->withoutFlags) != 0)
        continue;

      ret.push_back(sec);
    }

    sortInputSections(
        MutableArrayRef<InputSectionBase *>(ret).slice(sizeBefore), pat);
  }
  return ret;
}

void LinkerScript::discard(InputSectionBase *s) {
  if (s == in.shStrTab || s == mainPart->relrDyn)
    error("discarding " + s->name + " section is not allowed");

  // You can discard .hash and .gnu.hash sections by linker scripts. Since
  // they are synthesized sections, we need to handle them differently than
  // other regular sections.
  if (s == mainPart->gnuHashTab)
    mainPart->gnuHashTab = nullptr;
  if (s == mainPart->hashTab)
    mainPart->hashTab = nullptr;

  s->markDead();
  s->parent = nullptr;
  for (InputSection *ds : s->dependentSections)
    discard(ds);
}

void LinkerScript::discardSynthetic(OutputSection &outCmd) {
  for (Partition &part : partitions) {
    if (!part.armExidx || !part.armExidx->isLive())
      continue;
    std::vector<InputSectionBase *> secs(part.armExidx->exidxSections.begin(),
                                         part.armExidx->exidxSections.end());
    for (BaseCommand *base : outCmd.sectionCommands)
      if (auto *cmd = dyn_cast<InputSectionDescription>(base)) {
        std::vector<InputSectionBase *> matches =
            computeInputSections(cmd, secs);
        for (InputSectionBase *s : matches)
          discard(s);
      }
  }
}

std::vector<InputSectionBase *>
LinkerScript::createInputSectionList(OutputSection &outCmd) {
  std::vector<InputSectionBase *> ret;

  for (BaseCommand *base : outCmd.sectionCommands) {
    if (auto *cmd = dyn_cast<InputSectionDescription>(base)) {
      cmd->sectionBases = computeInputSections(cmd, inputSections);
      for (InputSectionBase *s : cmd->sectionBases)
        s->parent = &outCmd;
      ret.insert(ret.end(), cmd->sectionBases.begin(), cmd->sectionBases.end());
    }
  }
  return ret;
}

// Create output sections described by SECTIONS commands.
void LinkerScript::processSectionCommands() {
  size_t i = 0;
  for (BaseCommand *base : sectionCommands) {
    if (auto *sec = dyn_cast<OutputSection>(base)) {
      std::vector<InputSectionBase *> v = createInputSectionList(*sec);

      // The output section name `/DISCARD/' is special.
      // Any input section assigned to it is discarded.
      if (sec->name == "/DISCARD/") {
        for (InputSectionBase *s : v)
          discard(s);
        discardSynthetic(*sec);
        sec->sectionCommands.clear();
        continue;
      }

      // This is for ONLY_IF_RO and ONLY_IF_RW. An output section directive
      // ".foo : ONLY_IF_R[OW] { ... }" is handled only if all member input
      // sections satisfy a given constraint. If not, a directive is handled
      // as if it wasn't present from the beginning.
      //
      // Because we'll iterate over SectionCommands many more times, the easy
      // way to "make it as if it wasn't present" is to make it empty.
      if (!matchConstraints(v, sec->constraint)) {
        for (InputSectionBase *s : v)
          s->parent = nullptr;
        sec->sectionCommands.clear();
        continue;
      }

      // Handle subalign (e.g. ".foo : SUBALIGN(32) { ... }"). If subalign
      // is given, input sections are aligned to that value, whether the
      // given value is larger or smaller than the original section alignment.
      if (sec->subalignExpr) {
        uint32_t subalign = sec->subalignExpr().getValue();
        for (InputSectionBase *s : v)
          s->alignment = subalign;
      }

      // Set the partition field the same way OutputSection::recordSection()
      // does. Partitions cannot be used with the SECTIONS command, so this is
      // always 1.
      sec->partition = 1;

      sec->sectionIndex = i++;
    }
  }
}

void LinkerScript::processSymbolAssignments() {
  // Dot outside an output section still represents a relative address, whose
  // sh_shndx should not be SHN_UNDEF or SHN_ABS. Create a dummy aether section
  // that fills the void outside a section. It has an index of one, which is
  // indistinguishable from any other regular section index.
  aether = make<OutputSection>("", 0, SHF_ALLOC);
  aether->sectionIndex = 1;

  // ctx captures the local AddressState and makes it accessible deliberately.
  // This is needed as there are some cases where we cannot just thread the
  // current state through to a lambda function created by the script parser.
  AddressState state;
  ctx = &state;
  ctx->outSec = aether;

  for (BaseCommand *base : sectionCommands) {
    if (auto *cmd = dyn_cast<SymbolAssignment>(base))
      addSymbol(cmd);
    else
      for (BaseCommand *sub_base : cast<OutputSection>(base)->sectionCommands)
        if (auto *cmd = dyn_cast<SymbolAssignment>(sub_base))
          addSymbol(cmd);
  }

  ctx = nullptr;
}

static OutputSection *findByName(ArrayRef<BaseCommand *> vec,
                                 StringRef name) {
  for (BaseCommand *base : vec)
    if (auto *sec = dyn_cast<OutputSection>(base))
      if (sec->name == name)
        return sec;
  return nullptr;
}

static OutputSection *createSection(InputSectionBase *isec,
                                    StringRef outsecName) {
  OutputSection *sec = script->createOutputSection(outsecName, "<internal>");
  sec->recordSection(isec);
  return sec;
}

static OutputSection *
addInputSec(StringMap<TinyPtrVector<OutputSection *>> &map,
            InputSectionBase *isec, StringRef outsecName) {
  // Sections with SHT_GROUP or SHF_GROUP attributes reach here only when the -r
  // option is given. A section with SHT_GROUP defines a "section group", and
  // its members have SHF_GROUP attribute. Usually these flags have already been
  // stripped by InputFiles.cpp as section groups are processed and uniquified.
  // However, for the -r option, we want to pass through all section groups
  // as-is because adding/removing members or merging them with other groups
  // change their semantics.
  if (isec->type == SHT_GROUP || (isec->flags & SHF_GROUP))
    return createSection(isec, outsecName);

  // Imagine .zed : { *(.foo) *(.bar) } script. Both foo and bar may have
  // relocation sections .rela.foo and .rela.bar for example. Most tools do
  // not allow multiple REL[A] sections for output section. Hence we
  // should combine these relocation sections into single output.
  // We skip synthetic sections because it can be .rela.dyn/.rela.plt or any
  // other REL[A] sections created by linker itself.
  if (!isa<SyntheticSection>(isec) &&
      (isec->type == SHT_REL || isec->type == SHT_RELA)) {
    auto *sec = cast<InputSection>(isec);
    OutputSection *out = sec->getRelocatedSection()->getOutputSection();

    if (out->relocationSection) {
      out->relocationSection->recordSection(sec);
      return nullptr;
    }

    out->relocationSection = createSection(isec, outsecName);
    return out->relocationSection;
  }

  //  The ELF spec just says
  // ----------------------------------------------------------------
  // In the first phase, input sections that match in name, type and
  // attribute flags should be concatenated into single sections.
  // ----------------------------------------------------------------
  //
  // However, it is clear that at least some flags have to be ignored for
  // section merging. At the very least SHF_GROUP and SHF_COMPRESSED have to be
  // ignored. We should not have two output .text sections just because one was
  // in a group and another was not for example.
  //
  // It also seems that wording was a late addition and didn't get the
  // necessary scrutiny.
  //
  // Merging sections with different flags is expected by some users. One
  // reason is that if one file has
  //
  // int *const bar __attribute__((section(".foo"))) = (int *)0;
  //
  // gcc with -fPIC will produce a read only .foo section. But if another
  // file has
  //
  // int zed;
  // int *const bar __attribute__((section(".foo"))) = (int *)&zed;
  //
  // gcc with -fPIC will produce a read write section.
  //
  // Last but not least, when using linker script the merge rules are forced by
  // the script. Unfortunately, linker scripts are name based. This means that
  // expressions like *(.foo*) can refer to multiple input sections with
  // different flags. We cannot put them in different output sections or we
  // would produce wrong results for
  //
  // start = .; *(.foo.*) end = .; *(.bar)
  //
  // and a mapping of .foo1 and .bar1 to one section and .foo2 and .bar2 to
  // another. The problem is that there is no way to layout those output
  // sections such that the .foo sections are the only thing between the start
  // and end symbols.
  //
  // Given the above issues, we instead merge sections by name and error on
  // incompatible types and flags.
  TinyPtrVector<OutputSection *> &v = map[outsecName];
  for (OutputSection *sec : v) {
    if (sec->partition != isec->partition)
      continue;

    if (config->relocatable && (isec->flags & SHF_LINK_ORDER)) {
      // Merging two SHF_LINK_ORDER sections with different sh_link fields will
      // change their semantics, so we only merge them in -r links if they will
      // end up being linked to the same output section. The casts are fine
      // because everything in the map was created by the orphan placement code.
      auto *firstIsec = cast<InputSectionBase>(
          cast<InputSectionDescription>(sec->sectionCommands[0])
              ->sectionBases[0]);
      if (firstIsec->getLinkOrderDep()->getOutputSection() !=
          isec->getLinkOrderDep()->getOutputSection())
        continue;
    }

    sec->recordSection(isec);
    return nullptr;
  }

  OutputSection *sec = createSection(isec, outsecName);
  v.push_back(sec);
  return sec;
}

// Add sections that didn't match any sections command.
void LinkerScript::addOrphanSections() {
  StringMap<TinyPtrVector<OutputSection *>> map;
  std::vector<OutputSection *> v;

  std::function<void(InputSectionBase *)> add;
  add = [&](InputSectionBase *s) {
    if (s->isLive() && !s->parent) {
      orphanSections.push_back(s);

      StringRef name = getOutputSectionName(s);
      if (config->unique) {
        v.push_back(createSection(s, name));
      } else if (OutputSection *sec = findByName(sectionCommands, name)) {
        sec->recordSection(s);
      } else {
        if (OutputSection *os = addInputSec(map, s, name))
          v.push_back(os);
        assert(isa<MergeInputSection>(s) ||
               s->getOutputSection()->sectionIndex == UINT32_MAX);
      }
    }

    if (config->relocatable)
      for (InputSectionBase *depSec : s->dependentSections)
        if (depSec->flags & SHF_LINK_ORDER)
          add(depSec);
  };

  // For futher --emit-reloc handling code we need target output section
  // to be created before we create relocation output section, so we want
  // to create target sections first. We do not want priority handling
  // for synthetic sections because them are special.
  for (InputSectionBase *isec : inputSections) {
    // In -r links, SHF_LINK_ORDER sections are added while adding their parent
    // sections because we need to know the parent's output section before we
    // can select an output section for the SHF_LINK_ORDER section.
    if (config->relocatable && (isec->flags & SHF_LINK_ORDER))
      continue;

    if (auto *sec = dyn_cast<InputSection>(isec))
      if (InputSectionBase *rel = sec->getRelocatedSection())
        if (auto *relIS = dyn_cast_or_null<InputSectionBase>(rel->parent))
          add(relIS);
    add(isec);
  }

  // If no SECTIONS command was given, we should insert sections commands
  // before others, so that we can handle scripts which refers them,
  // for example: "foo = ABSOLUTE(ADDR(.text)));".
  // When SECTIONS command is present we just add all orphans to the end.
  if (hasSectionsCommand)
    sectionCommands.insert(sectionCommands.end(), v.begin(), v.end());
  else
    sectionCommands.insert(sectionCommands.begin(), v.begin(), v.end());
}

void LinkerScript::diagnoseOrphanHandling() const {
  for (const InputSectionBase *sec : orphanSections) {
    // Input SHT_REL[A] retained by --emit-relocs are ignored by
    // computeInputSections(). Don't warn/error.
    if (isa<InputSection>(sec) &&
        cast<InputSection>(sec)->getRelocatedSection())
      continue;

    StringRef name = getOutputSectionName(sec);
    if (config->orphanHandling == OrphanHandlingPolicy::Error)
      error(toString(sec) + " is being placed in '" + name + "'");
    else if (config->orphanHandling == OrphanHandlingPolicy::Warn)
      warn(toString(sec) + " is being placed in '" + name + "'");
  }
}

uint64_t LinkerScript::advance(uint64_t size, unsigned alignment) {
  bool isTbss =
      (ctx->outSec->flags & SHF_TLS) && ctx->outSec->type == SHT_NOBITS;
  uint64_t start = isTbss ? dot + ctx->threadBssOffset : dot;
  start = alignTo(start, alignment);
  uint64_t end = start + size;

  if (isTbss)
    ctx->threadBssOffset = end - dot;
  else
    dot = end;
  return end;
}

void LinkerScript::output(InputSection *s) {
  assert(ctx->outSec == s->getParent());
  uint64_t before = advance(0, 1);
  uint64_t pos = advance(s->getSize(), s->alignment);
  s->outSecOff = pos - s->getSize() - ctx->outSec->addr;

  // Update output section size after adding each section. This is so that
  // SIZEOF works correctly in the case below:
  // .foo { *(.aaa) a = SIZEOF(.foo); *(.bbb) }
  expandOutputSection(pos - before);
}

void LinkerScript::switchTo(OutputSection *sec) {
  ctx->outSec = sec;

  uint64_t pos = advance(0, 1);
  if (sec->addrExpr && script->hasSectionsCommand) {
    // The alignment is ignored.
    ctx->outSec->addr = pos;
  } else {
    // ctx->outSec->alignment is the max of ALIGN and the maximum of input
    // section alignments.
    ctx->outSec->addr = advance(0, ctx->outSec->alignment);
    expandMemoryRegions(ctx->outSec->addr - pos);
  }
}

// This function searches for a memory region to place the given output
// section in. If found, a pointer to the appropriate memory region is
// returned. Otherwise, a nullptr is returned.
MemoryRegion *LinkerScript::findMemoryRegion(OutputSection *sec) {
  // If a memory region name was specified in the output section command,
  // then try to find that region first.
  if (!sec->memoryRegionName.empty()) {
    if (MemoryRegion *m = memoryRegions.lookup(sec->memoryRegionName))
      return m;
    error("memory region '" + sec->memoryRegionName + "' not declared");
    return nullptr;
  }

  // If at least one memory region is defined, all sections must
  // belong to some memory region. Otherwise, we don't need to do
  // anything for memory regions.
  if (memoryRegions.empty())
    return nullptr;

  // See if a region can be found by matching section flags.
  for (auto &pair : memoryRegions) {
    MemoryRegion *m = pair.second;
    if ((m->flags & sec->flags) && (m->negFlags & sec->flags) == 0)
      return m;
  }

  // Otherwise, no suitable region was found.
  if (sec->flags & SHF_ALLOC)
    error("no memory region specified for section '" + sec->name + "'");
  return nullptr;
}

static OutputSection *findFirstSection(PhdrEntry *load) {
  for (OutputSection *sec : outputSections)
    if (sec->ptLoad == load)
      return sec;
  return nullptr;
}

// This function assigns offsets to input sections and an output section
// for a single sections command (e.g. ".text { *(.text); }").
void LinkerScript::assignOffsets(OutputSection *sec) {
  if (!(sec->flags & SHF_ALLOC))
    dot = 0;

  const bool sameMemRegion = ctx->memRegion == sec->memRegion;
  const bool prevLMARegionIsDefault = ctx->lmaRegion == nullptr;
  ctx->memRegion = sec->memRegion;
  ctx->lmaRegion = sec->lmaRegion;
  if (ctx->memRegion)
    dot = ctx->memRegion->curPos;

  if ((sec->flags & SHF_ALLOC) && sec->addrExpr)
    setDot(sec->addrExpr, sec->location, false);

  // If the address of the section has been moved forward by an explicit
  // expression so that it now starts past the current curPos of the enclosing
  // region, we need to expand the current region to account for the space
  // between the previous section, if any, and the start of this section.
  if (ctx->memRegion && ctx->memRegion->curPos < dot)
    expandMemoryRegion(ctx->memRegion, dot - ctx->memRegion->curPos,
                       ctx->memRegion->name, sec->name);

  switchTo(sec);

  // ctx->lmaOffset is LMA minus VMA. If LMA is explicitly specified via AT() or
  // AT>, recompute ctx->lmaOffset; otherwise, if both previous/current LMA
  // region is the default, and the two sections are in the same memory region,
  // reuse previous lmaOffset; otherwise, reset lmaOffset to 0. This emulates
  // heuristics described in
  // https://sourceware.org/binutils/docs/ld/Output-Section-LMA.html
  if (sec->lmaExpr)
    ctx->lmaOffset = sec->lmaExpr().getValue() - dot;
  else if (MemoryRegion *mr = sec->lmaRegion)
    ctx->lmaOffset = alignTo(mr->curPos, sec->alignment) - dot;
  else if (!sameMemRegion || !prevLMARegionIsDefault)
    ctx->lmaOffset = 0;

  // Propagate ctx->lmaOffset to the first "non-header" section.
  if (PhdrEntry *l = ctx->outSec->ptLoad)
    if (sec == findFirstSection(l))
      l->lmaOffset = ctx->lmaOffset;

  // We can call this method multiple times during the creation of
  // thunks and want to start over calculation each time.
  sec->size = 0;

  // We visited SectionsCommands from processSectionCommands to
  // layout sections. Now, we visit SectionsCommands again to fix
  // section offsets.
  for (BaseCommand *base : sec->sectionCommands) {
    // This handles the assignments to symbol or to the dot.
    if (auto *cmd = dyn_cast<SymbolAssignment>(base)) {
      cmd->addr = dot;
      assignSymbol(cmd, true);
      cmd->size = dot - cmd->addr;
      continue;
    }

    // Handle BYTE(), SHORT(), LONG(), or QUAD().
    if (auto *cmd = dyn_cast<ByteCommand>(base)) {
      cmd->offset = dot - ctx->outSec->addr;
      dot += cmd->size;
      expandOutputSection(cmd->size);
      continue;
    }

    // Handle a single input section description command.
    // It calculates and assigns the offsets for each section and also
    // updates the output section size.
    for (InputSection *sec : cast<InputSectionDescription>(base)->sections)
      output(sec);
  }
}

static bool isDiscardable(OutputSection &sec) {
  if (sec.name == "/DISCARD/")
    return true;

  // We do not remove empty sections that are explicitly
  // assigned to any segment.
  if (!sec.phdrs.empty())
    return false;

  // We do not want to remove OutputSections with expressions that reference
  // symbols even if the OutputSection is empty. We want to ensure that the
  // expressions can be evaluated and report an error if they cannot.
  if (sec.expressionsUseSymbols)
    return false;

  // OutputSections may be referenced by name in ADDR and LOADADDR expressions,
  // as an empty Section can has a valid VMA and LMA we keep the OutputSection
  // to maintain the integrity of the other Expression.
  if (sec.usedInExpression)
    return false;

  for (BaseCommand *base : sec.sectionCommands) {
    if (auto cmd = dyn_cast<SymbolAssignment>(base))
      // Don't create empty output sections just for unreferenced PROVIDE
      // symbols.
      if (cmd->name != "." && !cmd->sym)
        continue;

    if (!isa<InputSectionDescription>(*base))
      return false;
  }
  return true;
}

void LinkerScript::adjustSectionsBeforeSorting() {
  // If the output section contains only symbol assignments, create a
  // corresponding output section. The issue is what to do with linker script
  // like ".foo : { symbol = 42; }". One option would be to convert it to
  // "symbol = 42;". That is, move the symbol out of the empty section
  // description. That seems to be what bfd does for this simple case. The
  // problem is that this is not completely general. bfd will give up and
  // create a dummy section too if there is a ". = . + 1" inside the section
  // for example.
  // Given that we want to create the section, we have to worry what impact
  // it will have on the link. For example, if we just create a section with
  // 0 for flags, it would change which PT_LOADs are created.
  // We could remember that particular section is dummy and ignore it in
  // other parts of the linker, but unfortunately there are quite a few places
  // that would need to change:
  //   * The program header creation.
  //   * The orphan section placement.
  //   * The address assignment.
  // The other option is to pick flags that minimize the impact the section
  // will have on the rest of the linker. That is why we copy the flags from
  // the previous sections. Only a few flags are needed to keep the impact low.
  uint64_t flags = SHF_ALLOC;

  for (BaseCommand *&cmd : sectionCommands) {
    auto *sec = dyn_cast<OutputSection>(cmd);
    if (!sec)
      continue;

    // Handle align (e.g. ".foo : ALIGN(16) { ... }").
    if (sec->alignExpr)
      sec->alignment =
          std::max<uint32_t>(sec->alignment, sec->alignExpr().getValue());

    // The input section might have been removed (if it was an empty synthetic
    // section), but we at least know the flags.
    if (sec->hasInputSections)
      flags = sec->flags;

    // We do not want to keep any special flags for output section
    // in case it is empty.
    bool isEmpty = (getFirstInputSection(sec) == nullptr);
    if (isEmpty)
      sec->flags = flags & ((sec->nonAlloc ? 0 : (uint64_t)SHF_ALLOC) |
                            SHF_WRITE | SHF_EXECINSTR);

    if (isEmpty && isDiscardable(*sec)) {
      sec->markDead();
      cmd = nullptr;
    }
  }

  // It is common practice to use very generic linker scripts. So for any
  // given run some of the output sections in the script will be empty.
  // We could create corresponding empty output sections, but that would
  // clutter the output.
  // We instead remove trivially empty sections. The bfd linker seems even
  // more aggressive at removing them.
  llvm::erase_if(sectionCommands, [&](BaseCommand *base) { return !base; });
}

void LinkerScript::adjustSectionsAfterSorting() {
  // Try and find an appropriate memory region to assign offsets in.
  for (BaseCommand *base : sectionCommands) {
    if (auto *sec = dyn_cast<OutputSection>(base)) {
      if (!sec->lmaRegionName.empty()) {
        if (MemoryRegion *m = memoryRegions.lookup(sec->lmaRegionName))
          sec->lmaRegion = m;
        else
          error("memory region '" + sec->lmaRegionName + "' not declared");
      }
      sec->memRegion = findMemoryRegion(sec);
    }
  }

  // If output section command doesn't specify any segments,
  // and we haven't previously assigned any section to segment,
  // then we simply assign section to the very first load segment.
  // Below is an example of such linker script:
  // PHDRS { seg PT_LOAD; }
  // SECTIONS { .aaa : { *(.aaa) } }
  std::vector<StringRef> defPhdrs;
  auto firstPtLoad = llvm::find_if(phdrsCommands, [](const PhdrsCommand &cmd) {
    return cmd.type == PT_LOAD;
  });
  if (firstPtLoad != phdrsCommands.end())
    defPhdrs.push_back(firstPtLoad->name);

  // Walk the commands and propagate the program headers to commands that don't
  // explicitly specify them.
  for (BaseCommand *base : sectionCommands) {
    auto *sec = dyn_cast<OutputSection>(base);
    if (!sec)
      continue;

    if (sec->phdrs.empty()) {
      // To match the bfd linker script behaviour, only propagate program
      // headers to sections that are allocated.
      if (sec->flags & SHF_ALLOC)
        sec->phdrs = defPhdrs;
    } else {
      defPhdrs = sec->phdrs;
    }
  }
}

static uint64_t computeBase(uint64_t min, bool allocateHeaders) {
  // If there is no SECTIONS or if the linkerscript is explicit about program
  // headers, do our best to allocate them.
  if (!script->hasSectionsCommand || allocateHeaders)
    return 0;
  // Otherwise only allocate program headers if that would not add a page.
  return alignDown(min, config->maxPageSize);
}

// When the SECTIONS command is used, try to find an address for the file and
// program headers output sections, which can be added to the first PT_LOAD
// segment when program headers are created.
//
// We check if the headers fit below the first allocated section. If there isn't
// enough space for these sections, we'll remove them from the PT_LOAD segment,
// and we'll also remove the PT_PHDR segment.
void LinkerScript::allocateHeaders(std::vector<PhdrEntry *> &phdrs) {
  uint64_t min = std::numeric_limits<uint64_t>::max();
  for (OutputSection *sec : outputSections)
    if (sec->flags & SHF_ALLOC)
      min = std::min<uint64_t>(min, sec->addr);

  auto it = llvm::find_if(
      phdrs, [](const PhdrEntry *e) { return e->p_type == PT_LOAD; });
  if (it == phdrs.end())
    return;
  PhdrEntry *firstPTLoad = *it;

  bool hasExplicitHeaders =
      llvm::any_of(phdrsCommands, [](const PhdrsCommand &cmd) {
        return cmd.hasPhdrs || cmd.hasFilehdr;
      });
  bool paged = !config->omagic && !config->nmagic;
  uint64_t headerSize = getHeaderSize();
  if ((paged || hasExplicitHeaders) &&
      headerSize <= min - computeBase(min, hasExplicitHeaders)) {
    min = alignDown(min - headerSize, config->maxPageSize);
    Out::elfHeader->addr = min;
    Out::programHeaders->addr = min + Out::elfHeader->size;
    return;
  }

  // Error if we were explicitly asked to allocate headers.
  if (hasExplicitHeaders)
    error("could not allocate headers");

  Out::elfHeader->ptLoad = nullptr;
  Out::programHeaders->ptLoad = nullptr;
  firstPTLoad->firstSec = findFirstSection(firstPTLoad);

  llvm::erase_if(phdrs,
                 [](const PhdrEntry *e) { return e->p_type == PT_PHDR; });
}

LinkerScript::AddressState::AddressState() {
  for (auto &mri : script->memoryRegions) {
    MemoryRegion *mr = mri.second;
    mr->curPos = (mr->origin)().getValue();
  }
}

// Here we assign addresses as instructed by linker script SECTIONS
// sub-commands. Doing that allows us to use final VA values, so here
// we also handle rest commands like symbol assignments and ASSERTs.
// Returns a symbol that has changed its section or value, or nullptr if no
// symbol has changed.
const Defined *LinkerScript::assignAddresses() {
  if (script->hasSectionsCommand) {
    // With a linker script, assignment of addresses to headers is covered by
    // allocateHeaders().
    dot = config->imageBase.getValueOr(0);
  } else {
    // Assign addresses to headers right now.
    dot = target->getImageBase();
    Out::elfHeader->addr = dot;
    Out::programHeaders->addr = dot + Out::elfHeader->size;
    dot += getHeaderSize();
  }

  auto deleter = std::make_unique<AddressState>();
  ctx = deleter.get();
  errorOnMissingSection = true;
  switchTo(aether);

  SymbolAssignmentMap oldValues = getSymbolAssignmentValues(sectionCommands);
  for (BaseCommand *base : sectionCommands) {
    if (auto *cmd = dyn_cast<SymbolAssignment>(base)) {
      cmd->addr = dot;
      assignSymbol(cmd, false);
      cmd->size = dot - cmd->addr;
      continue;
    }
    assignOffsets(cast<OutputSection>(base));
  }

  ctx = nullptr;
  return getChangedSymbolAssignment(oldValues);
}

// Creates program headers as instructed by PHDRS linker script command.
std::vector<PhdrEntry *> LinkerScript::createPhdrs() {
  std::vector<PhdrEntry *> ret;

  // Process PHDRS and FILEHDR keywords because they are not
  // real output sections and cannot be added in the following loop.
  for (const PhdrsCommand &cmd : phdrsCommands) {
    PhdrEntry *phdr = make<PhdrEntry>(cmd.type, cmd.flags ? *cmd.flags : PF_R);

    if (cmd.hasFilehdr)
      phdr->add(Out::elfHeader);
    if (cmd.hasPhdrs)
      phdr->add(Out::programHeaders);

    if (cmd.lmaExpr) {
      phdr->p_paddr = cmd.lmaExpr().getValue();
      phdr->hasLMA = true;
    }
    ret.push_back(phdr);
  }

  // Add output sections to program headers.
  for (OutputSection *sec : outputSections) {
    // Assign headers specified by linker script
    for (size_t id : getPhdrIndices(sec)) {
      ret[id]->add(sec);
      if (!phdrsCommands[id].flags.hasValue())
        ret[id]->p_flags |= sec->getPhdrFlags();
    }
  }
  return ret;
}

// Returns true if we should emit an .interp section.
//
// We usually do. But if PHDRS commands are given, and
// no PT_INTERP is there, there's no place to emit an
// .interp, so we don't do that in that case.
bool LinkerScript::needsInterpSection() {
  if (phdrsCommands.empty())
    return true;
  for (PhdrsCommand &cmd : phdrsCommands)
    if (cmd.type == PT_INTERP)
      return true;
  return false;
}

ExprValue LinkerScript::getSymbolValue(StringRef name, const Twine &loc) {
  if (name == ".") {
    if (ctx)
      return {ctx->outSec, false, dot - ctx->outSec->addr, loc};
    error(loc + ": unable to get location counter value");
    return 0;
  }

  if (Symbol *sym = symtab->find(name)) {
    if (auto *ds = dyn_cast<Defined>(sym))
      return {ds->section, false, ds->value, loc};
    if (isa<SharedSymbol>(sym))
      if (!errorOnMissingSection)
        return {nullptr, false, 0, loc};
  }

  error(loc + ": symbol not found: " + name);
  return 0;
}

// Returns the index of the segment named Name.
static Optional<size_t> getPhdrIndex(ArrayRef<PhdrsCommand> vec,
                                     StringRef name) {
  for (size_t i = 0; i < vec.size(); ++i)
    if (vec[i].name == name)
      return i;
  return None;
}

// Returns indices of ELF headers containing specific section. Each index is a
// zero based number of ELF header listed within PHDRS {} script block.
std::vector<size_t> LinkerScript::getPhdrIndices(OutputSection *cmd) {
  std::vector<size_t> ret;

  for (StringRef s : cmd->phdrs) {
    if (Optional<size_t> idx = getPhdrIndex(phdrsCommands, s))
      ret.push_back(*idx);
    else if (s != "NONE")
      error(cmd->location + ": program header '" + s +
            "' is not listed in PHDRS");
  }
  return ret;
}
