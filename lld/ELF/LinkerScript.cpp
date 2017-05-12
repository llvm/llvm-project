//===- LinkerScript.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the parser/evaluator of the linker script.
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "InputSection.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
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

LinkerScript *elf::Script;

uint64_t ExprValue::getValue() const {
  if (Sec) {
    if (Sec->getOutputSection())
      return Sec->getOffset(Val) + Sec->getOutputSection()->Addr;
    error("unable to evaluate expression: input section " + Sec->Name +
          " has no output section assigned");
  }
  return Val;
}

uint64_t ExprValue::getSecAddr() const {
  if (Sec)
    return Sec->getOffset(0) + Sec->getOutputSection()->Addr;
  return 0;
}

template <class ELFT> static SymbolBody *addRegular(SymbolAssignment *Cmd) {
  Symbol *Sym;
  uint8_t Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  std::tie(Sym, std::ignore) = Symtab<ELFT>::X->insert(
      Cmd->Name, /*Type*/ 0, Visibility, /*CanOmitFromDynSym*/ false,
      /*File*/ nullptr);
  Sym->Binding = STB_GLOBAL;
  ExprValue Value = Cmd->Expression();
  SectionBase *Sec = Value.isAbsolute() ? nullptr : Value.Sec;

  // We want to set symbol values early if we can. This allows us to use symbols
  // as variables in linker scripts. Doing so allows us to write expressions
  // like this: `alignment = 16; . = ALIGN(., alignment)`
  uint64_t SymValue = Value.isAbsolute() ? Value.getValue() : 0;
  replaceBody<DefinedRegular>(Sym, Cmd->Name, /*IsLocal=*/false, Visibility,
                              STT_NOTYPE, SymValue, 0, Sec, nullptr);
  return Sym->body();
}

OutputSection *LinkerScript::getOutputSection(const Twine &Loc,
                                              StringRef Name) {
  for (OutputSection *Sec : *OutputSections)
    if (Sec->Name == Name)
      return Sec;

  static OutputSection Dummy("", 0, 0);
  if (ErrorOnMissingSection)
    error(Loc + ": undefined section " + Name);
  return &Dummy;
}

// This function is essentially the same as getOutputSection(Name)->Size,
// but it won't print out an error message if a given section is not found.
//
// Linker script does not create an output section if its content is empty.
// We want to allow SIZEOF(.foo) where .foo is a section which happened to
// be empty. That is why this function is different from getOutputSection().
uint64_t LinkerScript::getOutputSectionSize(StringRef Name) {
  for (OutputSection *Sec : *OutputSections)
    if (Sec->Name == Name)
      return Sec->Size;
  return 0;
}

void LinkerScript::setDot(Expr E, const Twine &Loc, bool InSec) {
  uint64_t Val = E().getValue();
  if (Val < Dot) {
    if (InSec)
      error(Loc + ": unable to move location counter backward for: " +
            CurOutSec->Name);
    else
      error(Loc + ": unable to move location counter backward");
  }
  Dot = Val;
  // Update to location counter means update to section size.
  if (InSec)
    CurOutSec->Size = Dot - CurOutSec->Addr;
}

// Sets value of a symbol. Two kinds of symbols are processed: synthetic
// symbols, whose value is an offset from beginning of section and regular
// symbols whose value is absolute.
void LinkerScript::assignSymbol(SymbolAssignment *Cmd, bool InSec) {
  if (Cmd->Name == ".") {
    setDot(Cmd->Expression, Cmd->Location, InSec);
    return;
  }

  if (!Cmd->Sym)
    return;

  auto *Sym = cast<DefinedRegular>(Cmd->Sym);
  ExprValue V = Cmd->Expression();
  if (V.isAbsolute()) {
    Sym->Value = V.getValue();
  } else {
    Sym->Section = V.Sec;
    if (Sym->Section->Flags & SHF_ALLOC)
      Sym->Value = V.Val;
    else
      Sym->Value = V.getValue();
  }
}

static SymbolBody *findSymbol(StringRef S) {
  switch (Config->EKind) {
  case ELF32LEKind:
    return Symtab<ELF32LE>::X->find(S);
  case ELF32BEKind:
    return Symtab<ELF32BE>::X->find(S);
  case ELF64LEKind:
    return Symtab<ELF64LE>::X->find(S);
  case ELF64BEKind:
    return Symtab<ELF64BE>::X->find(S);
  default:
    llvm_unreachable("unknown Config->EKind");
  }
}

static SymbolBody *addRegularSymbol(SymbolAssignment *Cmd) {
  switch (Config->EKind) {
  case ELF32LEKind:
    return addRegular<ELF32LE>(Cmd);
  case ELF32BEKind:
    return addRegular<ELF32BE>(Cmd);
  case ELF64LEKind:
    return addRegular<ELF64LE>(Cmd);
  case ELF64BEKind:
    return addRegular<ELF64BE>(Cmd);
  default:
    llvm_unreachable("unknown Config->EKind");
  }
}

void LinkerScript::addSymbol(SymbolAssignment *Cmd) {
  if (Cmd->Name == ".")
    return;

  // If a symbol was in PROVIDE(), we need to define it only when
  // it is a referenced undefined symbol.
  SymbolBody *B = findSymbol(Cmd->Name);
  if (Cmd->Provide && (!B || B->isDefined()))
    return;

  Cmd->Sym = addRegularSymbol(Cmd);
}

bool SymbolAssignment::classof(const BaseCommand *C) {
  return C->Kind == AssignmentKind;
}

bool OutputSectionCommand::classof(const BaseCommand *C) {
  return C->Kind == OutputSectionKind;
}

bool InputSectionDescription::classof(const BaseCommand *C) {
  return C->Kind == InputSectionKind;
}

bool AssertCommand::classof(const BaseCommand *C) {
  return C->Kind == AssertKind;
}

bool BytesDataCommand::classof(const BaseCommand *C) {
  return C->Kind == BytesDataKind;
}

static StringRef basename(InputSectionBase *S) {
  if (S->File)
    return sys::path::filename(S->File->getName());
  return "";
}

bool LinkerScript::shouldKeep(InputSectionBase *S) {
  for (InputSectionDescription *ID : Opt.KeptSections)
    if (ID->FilePat.match(basename(S)))
      for (SectionPattern &P : ID->SectionPatterns)
        if (P.SectionPat.match(S->Name))
          return true;
  return false;
}

// A helper function for the SORT() command.
static std::function<bool(InputSectionBase *, InputSectionBase *)>
getComparator(SortSectionPolicy K) {
  switch (K) {
  case SortSectionPolicy::Alignment:
    return [](InputSectionBase *A, InputSectionBase *B) {
      // ">" is not a mistake. Sections with larger alignments are placed
      // before sections with smaller alignments in order to reduce the
      // amount of padding necessary. This is compatible with GNU.
      return A->Alignment > B->Alignment;
    };
  case SortSectionPolicy::Name:
    return [](InputSectionBase *A, InputSectionBase *B) {
      return A->Name < B->Name;
    };
  case SortSectionPolicy::Priority:
    return [](InputSectionBase *A, InputSectionBase *B) {
      return getPriority(A->Name) < getPriority(B->Name);
    };
  default:
    llvm_unreachable("unknown sort policy");
  }
}

// A helper function for the SORT() command.
static bool matchConstraints(ArrayRef<InputSectionBase *> Sections,
                             ConstraintKind Kind) {
  if (Kind == ConstraintKind::NoConstraint)
    return true;

  bool IsRW = llvm::any_of(Sections, [](InputSectionBase *Sec) {
    return static_cast<InputSectionBase *>(Sec)->Flags & SHF_WRITE;
  });

  return (IsRW && Kind == ConstraintKind::ReadWrite) ||
         (!IsRW && Kind == ConstraintKind::ReadOnly);
}

static void sortSections(InputSectionBase **Begin, InputSectionBase **End,
                         SortSectionPolicy K) {
  if (K != SortSectionPolicy::Default && K != SortSectionPolicy::None)
    std::stable_sort(Begin, End, getComparator(K));
}

// Compute and remember which sections the InputSectionDescription matches.
std::vector<InputSectionBase *>
LinkerScript::computeInputSections(const InputSectionDescription *Cmd) {
  std::vector<InputSectionBase *> Ret;

  // Collects all sections that satisfy constraints of Cmd.
  for (const SectionPattern &Pat : Cmd->SectionPatterns) {
    size_t SizeBefore = Ret.size();

    for (InputSectionBase *Sec : InputSections) {
      if (Sec->Assigned)
        continue;

      // For -emit-relocs we have to ignore entries like
      //   .rela.dyn : { *(.rela.data) }
      // which are common because they are in the default bfd script.
      if (Sec->Type == SHT_REL || Sec->Type == SHT_RELA)
        continue;

      StringRef Filename = basename(Sec);
      if (!Cmd->FilePat.match(Filename) ||
          Pat.ExcludedFilePat.match(Filename) ||
          !Pat.SectionPat.match(Sec->Name))
        continue;

      Ret.push_back(Sec);
      Sec->Assigned = true;
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
    InputSectionBase **Begin = Ret.data() + SizeBefore;
    InputSectionBase **End = Ret.data() + Ret.size();
    if (Pat.SortOuter != SortSectionPolicy::None) {
      if (Pat.SortInner == SortSectionPolicy::Default)
        sortSections(Begin, End, Config->SortSection);
      else
        sortSections(Begin, End, Pat.SortInner);
      sortSections(Begin, End, Pat.SortOuter);
    }
  }
  return Ret;
}

void LinkerScript::discard(ArrayRef<InputSectionBase *> V) {
  for (InputSectionBase *S : V) {
    S->Live = false;
    if (S == InX::ShStrTab)
      error("discarding .shstrtab section is not allowed");
    discard(S->DependentSections);
  }
}

std::vector<InputSectionBase *>
LinkerScript::createInputSectionList(OutputSectionCommand &OutCmd) {
  std::vector<InputSectionBase *> Ret;

  for (BaseCommand *Base : OutCmd.Commands) {
    auto *Cmd = dyn_cast<InputSectionDescription>(Base);
    if (!Cmd)
      continue;

    Cmd->Sections = computeInputSections(Cmd);
    Ret.insert(Ret.end(), Cmd->Sections.begin(), Cmd->Sections.end());
  }

  return Ret;
}

void LinkerScript::processCommands(OutputSectionFactory &Factory) {
  // A symbol can be assigned before any section is mentioned in the linker
  // script. In an DSO, the symbol values are addresses, so the only important
  // section values are:
  // * SHN_UNDEF
  // * SHN_ABS
  // * Any value meaning a regular section.
  // To handle that, create a dummy aether section that fills the void before
  // the linker scripts switches to another section. It has an index of one
  // which will map to whatever the first actual section is.
  Aether = make<OutputSection>("", 0, SHF_ALLOC);
  Aether->SectionIndex = 1;
  CurOutSec = Aether;
  Dot = 0;

  for (size_t I = 0; I < Opt.Commands.size(); ++I) {
    // Handle symbol assignments outside of any output section.
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Opt.Commands[I])) {
      addSymbol(Cmd);
      continue;
    }

    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Opt.Commands[I])) {
      std::vector<InputSectionBase *> V = createInputSectionList(*Cmd);

      // The output section name `/DISCARD/' is special.
      // Any input section assigned to it is discarded.
      if (Cmd->Name == "/DISCARD/") {
        discard(V);
        continue;
      }

      // This is for ONLY_IF_RO and ONLY_IF_RW. An output section directive
      // ".foo : ONLY_IF_R[OW] { ... }" is handled only if all member input
      // sections satisfy a given constraint. If not, a directive is handled
      // as if it wasn't present from the beginning.
      //
      // Because we'll iterate over Commands many more times, the easiest
      // way to "make it as if it wasn't present" is to just remove it.
      if (!matchConstraints(V, Cmd->Constraint)) {
        for (InputSectionBase *S : V)
          S->Assigned = false;
        Opt.Commands.erase(Opt.Commands.begin() + I);
        --I;
        continue;
      }

      // A directive may contain symbol definitions like this:
      // ".foo : { ...; bar = .; }". Handle them.
      for (BaseCommand *Base : Cmd->Commands)
        if (auto *OutCmd = dyn_cast<SymbolAssignment>(Base))
          addSymbol(OutCmd);

      // Handle subalign (e.g. ".foo : SUBALIGN(32) { ... }"). If subalign
      // is given, input sections are aligned to that value, whether the
      // given value is larger or smaller than the original section alignment.
      if (Cmd->SubalignExpr) {
        uint32_t Subalign = Cmd->SubalignExpr().getValue();
        for (InputSectionBase *S : V)
          S->Alignment = Subalign;
      }

      // Add input sections to an output section.
      for (InputSectionBase *S : V)
        Factory.addInputSec(S, Cmd->Name, Cmd->Sec);
      if (OutputSection *Sec = Cmd->Sec) {
        assert(Sec->SectionIndex == INT_MAX);
        Sec->SectionIndex = I;
        SecToCommand[Sec] = Cmd;
      }
    }
  }
  CurOutSec = nullptr;
}

void LinkerScript::fabricateDefaultCommands() {
  std::vector<BaseCommand *> Commands;

  // Define start address
  uint64_t StartAddr = Config->ImageBase + elf::getHeaderSize();

  // The Sections with -T<section> have been sorted in order of ascending
  // address. We must lower StartAddr if the lowest -T<section address> as
  // calls to setDot() must be monotonically increasing.
  for (auto& KV : Config->SectionStartMap)
    StartAddr = std::min(StartAddr, KV.second);

  Commands.push_back(
      make<SymbolAssignment>(".", [=] { return StartAddr; }, ""));

  // For each OutputSection that needs a VA fabricate an OutputSectionCommand
  // with an InputSectionDescription describing the InputSections
  for (OutputSection *Sec : *OutputSections) {
    if (!(Sec->Flags & SHF_ALLOC))
      continue;

    auto *OSCmd = make<OutputSectionCommand>(Sec->Name);
    OSCmd->Sec = Sec;
    SecToCommand[Sec] = OSCmd;

    // Prefer user supplied address over additional alignment constraint
    auto I = Config->SectionStartMap.find(Sec->Name);
    if (I != Config->SectionStartMap.end())
      Commands.push_back(
          make<SymbolAssignment>(".", [=] { return I->second; }, ""));
    else if (Sec->PageAlign)
      OSCmd->AddrExpr = [=] {
        return alignTo(Script->getDot(), Config->MaxPageSize);
      };

    Commands.push_back(OSCmd);
    if (Sec->Sections.size()) {
      auto *ISD = make<InputSectionDescription>("");
      OSCmd->Commands.push_back(ISD);
      for (InputSection *ISec : Sec->Sections) {
        ISD->Sections.push_back(ISec);
        ISec->Assigned = true;
      }
    }
  }
  // SECTIONS commands run before other non SECTIONS commands
  Commands.insert(Commands.end(), Opt.Commands.begin(), Opt.Commands.end());
  Opt.Commands = std::move(Commands);
}

// Add sections that didn't match any sections command.
void LinkerScript::addOrphanSections(OutputSectionFactory &Factory) {
  for (InputSectionBase *S : InputSections) {
    if (!S->Live || S->OutSec)
      continue;
    StringRef Name = getOutputSectionName(S->Name);
    auto I = std::find_if(
        Opt.Commands.begin(), Opt.Commands.end(), [&](BaseCommand *Base) {
          if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base))
            return Cmd->Name == Name;
          return false;
        });
    if (I == Opt.Commands.end()) {
      Factory.addInputSec(S, Name);
    } else {
      auto *Cmd = cast<OutputSectionCommand>(*I);
      Factory.addInputSec(S, Name, Cmd->Sec);
      if (OutputSection *Sec = Cmd->Sec) {
        SecToCommand[Sec] = Cmd;
        unsigned Index = std::distance(Opt.Commands.begin(), I);
        assert(Sec->SectionIndex == INT_MAX || Sec->SectionIndex == Index);
        Sec->SectionIndex = Index;
      }
      auto *ISD = make<InputSectionDescription>("");
      ISD->Sections.push_back(S);
      Cmd->Commands.push_back(ISD);
    }
  }
}

uint64_t LinkerScript::advance(uint64_t Size, unsigned Align) {
  bool IsTbss = (CurOutSec->Flags & SHF_TLS) && CurOutSec->Type == SHT_NOBITS;
  uint64_t Start = IsTbss ? Dot + ThreadBssOffset : Dot;
  Start = alignTo(Start, Align);
  uint64_t End = Start + Size;

  if (IsTbss)
    ThreadBssOffset = End - Dot;
  else
    Dot = End;
  return End;
}

void LinkerScript::output(InputSection *S) {
  uint64_t Pos = advance(S->getSize(), S->Alignment);
  S->OutSecOff = Pos - S->getSize() - CurOutSec->Addr;

  // Update output section size after adding each section. This is so that
  // SIZEOF works correctly in the case below:
  // .foo { *(.aaa) a = SIZEOF(.foo); *(.bbb) }
  CurOutSec->Size = Pos - CurOutSec->Addr;

  // If there is a memory region associated with this input section, then
  // place the section in that region and update the region index.
  if (CurMemRegion) {
    CurMemRegion->Offset += CurOutSec->Size;
    uint64_t CurSize = CurMemRegion->Offset - CurMemRegion->Origin;
    if (CurSize > CurMemRegion->Length) {
      uint64_t OverflowAmt = CurSize - CurMemRegion->Length;
      error("section '" + CurOutSec->Name + "' will not fit in region '" +
            CurMemRegion->Name + "': overflowed by " + Twine(OverflowAmt) +
            " bytes");
    }
  }
}

void LinkerScript::switchTo(OutputSection *Sec) {
  if (CurOutSec == Sec)
    return;

  CurOutSec = Sec;
  CurOutSec->Addr = advance(0, CurOutSec->Alignment);

  // If neither AT nor AT> is specified for an allocatable section, the linker
  // will set the LMA such that the difference between VMA and LMA for the
  // section is the same as the preceding output section in the same region
  // https://sourceware.org/binutils/docs-2.20/ld/Output-Section-LMA.html
  if (LMAOffset)
    CurOutSec->LMAOffset = LMAOffset();
}

void LinkerScript::process(BaseCommand &Base) {
  // This handles the assignments to symbol or to the dot.
  if (auto *Cmd = dyn_cast<SymbolAssignment>(&Base)) {
    assignSymbol(Cmd, true);
    return;
  }

  // Handle BYTE(), SHORT(), LONG(), or QUAD().
  if (auto *Cmd = dyn_cast<BytesDataCommand>(&Base)) {
    Cmd->Offset = Dot - CurOutSec->Addr;
    Dot += Cmd->Size;
    CurOutSec->Size = Dot - CurOutSec->Addr;
    return;
  }

  // Handle ASSERT().
  if (auto *Cmd = dyn_cast<AssertCommand>(&Base)) {
    Cmd->Expression();
    return;
  }

  // Handle a single input section description command.
  // It calculates and assigns the offsets for each section and also
  // updates the output section size.
  auto &Cmd = cast<InputSectionDescription>(Base);
  for (InputSectionBase *Sec : Cmd.Sections) {
    // We tentatively added all synthetic sections at the beginning and removed
    // empty ones afterwards (because there is no way to know whether they were
    // going be empty or not other than actually running linker scripts.)
    // We need to ignore remains of empty sections.
    if (auto *S = dyn_cast<SyntheticSection>(Sec))
      if (S->empty())
        continue;

    if (!Sec->Live)
      continue;
    assert(CurOutSec == Sec->OutSec);
    output(cast<InputSection>(Sec));
  }
}

// This function searches for a memory region to place the given output
// section in. If found, a pointer to the appropriate memory region is
// returned. Otherwise, a nullptr is returned.
MemoryRegion *LinkerScript::findMemoryRegion(OutputSectionCommand *Cmd) {
  // If a memory region name was specified in the output section command,
  // then try to find that region first.
  if (!Cmd->MemoryRegionName.empty()) {
    auto It = Opt.MemoryRegions.find(Cmd->MemoryRegionName);
    if (It != Opt.MemoryRegions.end())
      return &It->second;
    error("memory region '" + Cmd->MemoryRegionName + "' not declared");
    return nullptr;
  }

  // If at least one memory region is defined, all sections must
  // belong to some memory region. Otherwise, we don't need to do
  // anything for memory regions.
  if (Opt.MemoryRegions.empty())
    return nullptr;

  OutputSection *Sec = Cmd->Sec;
  // See if a region can be found by matching section flags.
  for (auto &Pair : Opt.MemoryRegions) {
    MemoryRegion &M = Pair.second;
    if ((M.Flags & Sec->Flags) && (M.NegFlags & Sec->Flags) == 0)
      return &M;
  }

  // Otherwise, no suitable region was found.
  if (Sec->Flags & SHF_ALLOC)
    error("no memory region specified for section '" + Sec->Name + "'");
  return nullptr;
}

// This function assigns offsets to input sections and an output section
// for a single sections command (e.g. ".text { *(.text); }").
void LinkerScript::assignOffsets(OutputSectionCommand *Cmd) {
  OutputSection *Sec = Cmd->Sec;
  if (!Sec)
    return;

  if (Cmd->AddrExpr && (Sec->Flags & SHF_ALLOC))
    setDot(Cmd->AddrExpr, Cmd->Location, false);

  if (Cmd->LMAExpr) {
    uint64_t D = Dot;
    LMAOffset = [=] { return Cmd->LMAExpr().getValue() - D; };
  }

  CurMemRegion = Cmd->MemRegion;
  if (CurMemRegion)
    Dot = CurMemRegion->Offset;
  switchTo(Sec);

  // We do not support custom layout for compressed debug sectons.
  // At this point we already know their size and have compressed content.
  if (CurOutSec->Flags & SHF_COMPRESSED)
    return;

  for (BaseCommand *C : Cmd->Commands)
    process(*C);
}

void LinkerScript::removeEmptyCommands() {
  // It is common practice to use very generic linker scripts. So for any
  // given run some of the output sections in the script will be empty.
  // We could create corresponding empty output sections, but that would
  // clutter the output.
  // We instead remove trivially empty sections. The bfd linker seems even
  // more aggressive at removing them.
  auto Pos = std::remove_if(
      Opt.Commands.begin(), Opt.Commands.end(), [&](BaseCommand *Base) {
        if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base))
          return std::find(OutputSections->begin(), OutputSections->end(),
                           Cmd->Sec) == OutputSections->end();
        return false;
      });
  Opt.Commands.erase(Pos, Opt.Commands.end());
}

static bool isAllSectionDescription(const OutputSectionCommand &Cmd) {
  for (BaseCommand *Base : Cmd.Commands)
    if (!isa<InputSectionDescription>(*Base))
      return false;
  return true;
}

void LinkerScript::adjustSectionsBeforeSorting() {
  // If the output section contains only symbol assignments, create a
  // corresponding output section. The bfd linker seems to only create them if
  // '.' is assigned to, but creating these section should not have any bad
  // consequeces and gives us a section to put the symbol in.
  uint64_t Flags = SHF_ALLOC;
  uint32_t Type = SHT_PROGBITS;

  for (int I = 0, E = Opt.Commands.size(); I != E; ++I) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Opt.Commands[I]);
    if (!Cmd)
      continue;
    if (OutputSection *Sec = Cmd->Sec) {
      Flags = Sec->Flags;
      Type = Sec->Type;
      continue;
    }

    if (isAllSectionDescription(*Cmd))
      continue;

    auto *OutSec = make<OutputSection>(Cmd->Name, Type, Flags);
    OutSec->SectionIndex = I;
    OutputSections->push_back(OutSec);
    Cmd->Sec = OutSec;
    SecToCommand[OutSec] = Cmd;
  }
}

void LinkerScript::adjustSectionsAfterSorting() {
  placeOrphanSections();

  // Try and find an appropriate memory region to assign offsets in.
  for (BaseCommand *Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base)) {
      Cmd->MemRegion = findMemoryRegion(Cmd);
      // Handle align (e.g. ".foo : ALIGN(16) { ... }").
      if (Cmd->AlignExpr)
	Cmd->Sec->updateAlignment(Cmd->AlignExpr().getValue());
    }
  }

  // If output section command doesn't specify any segments,
  // and we haven't previously assigned any section to segment,
  // then we simply assign section to the very first load segment.
  // Below is an example of such linker script:
  // PHDRS { seg PT_LOAD; }
  // SECTIONS { .aaa : { *(.aaa) } }
  std::vector<StringRef> DefPhdrs;
  auto FirstPtLoad =
      std::find_if(Opt.PhdrsCommands.begin(), Opt.PhdrsCommands.end(),
                   [](const PhdrsCommand &Cmd) { return Cmd.Type == PT_LOAD; });
  if (FirstPtLoad != Opt.PhdrsCommands.end())
    DefPhdrs.push_back(FirstPtLoad->Name);

  // Walk the commands and propagate the program headers to commands that don't
  // explicitly specify them.
  for (BaseCommand *Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base);
    if (!Cmd)
      continue;

    if (Cmd->Phdrs.empty())
      Cmd->Phdrs = DefPhdrs;
    else
      DefPhdrs = Cmd->Phdrs;
  }

  removeEmptyCommands();
}

// When placing orphan sections, we want to place them after symbol assignments
// so that an orphan after
//   begin_foo = .;
//   foo : { *(foo) }
//   end_foo = .;
// doesn't break the intended meaning of the begin/end symbols.
// We don't want to go over sections since Writer<ELFT>::sortSections is the
// one in charge of deciding the order of the sections.
// We don't want to go over alignments, since doing so in
//  rx_sec : { *(rx_sec) }
//  . = ALIGN(0x1000);
//  /* The RW PT_LOAD starts here*/
//  rw_sec : { *(rw_sec) }
// would mean that the RW PT_LOAD would become unaligned.
static bool shouldSkip(BaseCommand *Cmd) {
  if (isa<OutputSectionCommand>(Cmd))
    return false;
  if (auto *Assign = dyn_cast<SymbolAssignment>(Cmd))
    return Assign->Name != ".";
  return true;
}

// Orphan sections are sections present in the input files which are
// not explicitly placed into the output file by the linker script.
//
// When the control reaches this function, Opt.Commands contains
// output section commands for non-orphan sections only. This function
// adds new elements for orphan sections so that all sections are
// explicitly handled by Opt.Commands.
//
// Writer<ELFT>::sortSections has already sorted output sections.
// What we need to do is to scan OutputSections vector and
// Opt.Commands in parallel to find orphan sections. If there is an
// output section that doesn't have a corresponding entry in
// Opt.Commands, we will insert a new entry to Opt.Commands.
//
// There is some ambiguity as to where exactly a new entry should be
// inserted, because Opt.Commands contains not only output section
// commands but also other types of commands such as symbol assignment
// expressions. There's no correct answer here due to the lack of the
// formal specification of the linker script. We use heuristics to
// determine whether a new output command should be added before or
// after another commands. For the details, look at shouldSkip
// function.
void LinkerScript::placeOrphanSections() {
  // The OutputSections are already in the correct order.
  // This loops creates or moves commands as needed so that they are in the
  // correct order.
  int CmdIndex = 0;

  // As a horrible special case, skip the first . assignment if it is before any
  // section. We do this because it is common to set a load address by starting
  // the script with ". = 0xabcd" and the expectation is that every section is
  // after that.
  auto FirstSectionOrDotAssignment =
      std::find_if(Opt.Commands.begin(), Opt.Commands.end(),
                   [](BaseCommand *Cmd) { return !shouldSkip(Cmd); });
  if (FirstSectionOrDotAssignment != Opt.Commands.end()) {
    CmdIndex = FirstSectionOrDotAssignment - Opt.Commands.begin();
    if (isa<SymbolAssignment>(**FirstSectionOrDotAssignment))
      ++CmdIndex;
  }

  for (OutputSection *Sec : *OutputSections) {
    StringRef Name = Sec->Name;

    // Find the last spot where we can insert a command and still get the
    // correct result.
    auto CmdIter = Opt.Commands.begin() + CmdIndex;
    auto E = Opt.Commands.end();
    while (CmdIter != E && shouldSkip(*CmdIter)) {
      ++CmdIter;
      ++CmdIndex;
    }

    // If there is no command corresponding to this output section,
    // create one and put a InputSectionDescription in it so that both
    // representations agree on which input sections to use.
    OutputSectionCommand *Cmd = getCmd(Sec);
    if (!Cmd) {
      Cmd = make<OutputSectionCommand>(Name);
      Opt.Commands.insert(CmdIter, Cmd);
      ++CmdIndex;

      Cmd->Sec = Sec;
      SecToCommand[Sec] = Cmd;
      auto *ISD = make<InputSectionDescription>("");
      for (InputSection *IS : Sec->Sections)
        ISD->Sections.push_back(IS);
      Cmd->Commands.push_back(ISD);

      continue;
    }

    // Continue from where we found it.
    while (*CmdIter != Cmd) {
      ++CmdIter;
      ++CmdIndex;
    }
    ++CmdIndex;
  }
}

void LinkerScript::processNonSectionCommands() {
  for (BaseCommand *Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base))
      assignSymbol(Cmd, false);
    else if (auto *Cmd = dyn_cast<AssertCommand>(Base))
      Cmd->Expression();
  }
}

// Do a last effort at synchronizing the linker script "AST" and the section
// list. This is needed to account for last minute changes, like adding a
// .ARM.exidx terminator and sorting SHF_LINK_ORDER sections.
//
// FIXME: We should instead create the "AST" earlier and the above changes would
// be done directly in the "AST".
//
// This can only handle new sections being added and sections being reordered.
void LinkerScript::synchronize() {
  for (BaseCommand *Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base);
    if (!Cmd)
      continue;
    ArrayRef<InputSection *> Sections = Cmd->Sec->Sections;
    std::vector<InputSectionBase **> ScriptSections;
    DenseSet<InputSectionBase *> ScriptSectionsSet;
    for (BaseCommand *Base : Cmd->Commands) {
      auto *ISD = dyn_cast<InputSectionDescription>(Base);
      if (!ISD)
        continue;
      for (InputSectionBase *&IS : ISD->Sections) {
        if (IS->Live) {
          ScriptSections.push_back(&IS);
          ScriptSectionsSet.insert(IS);
        }
      }
    }
    std::vector<InputSectionBase *> Missing;
    for (InputSection *IS : Sections)
      if (!ScriptSectionsSet.count(IS))
        Missing.push_back(IS);
    if (!Missing.empty()) {
      auto ISD = make<InputSectionDescription>("");
      ISD->Sections = Missing;
      Cmd->Commands.push_back(ISD);
      for (InputSectionBase *&IS : ISD->Sections)
        if (IS->Live)
          ScriptSections.push_back(&IS);
    }
    assert(ScriptSections.size() == Sections.size());
    for (int I = 0, N = Sections.size(); I < N; ++I)
      *ScriptSections[I] = Sections[I];
  }
}

static bool allocateHeaders(std::vector<PhdrEntry> &Phdrs,
                            ArrayRef<OutputSection *> OutputSections,
                            uint64_t Min) {
  auto FirstPTLoad =
      std::find_if(Phdrs.begin(), Phdrs.end(),
                   [](const PhdrEntry &E) { return E.p_type == PT_LOAD; });
  if (FirstPTLoad == Phdrs.end())
    return false;

  uint64_t HeaderSize = getHeaderSize();
  if (HeaderSize <= Min || Script->hasPhdrsCommands()) {
    Min = alignDown(Min - HeaderSize, Config->MaxPageSize);
    Out::ElfHeader->Addr = Min;
    Out::ProgramHeaders->Addr = Min + Out::ElfHeader->Size;
    return true;
  }

  assert(FirstPTLoad->First == Out::ElfHeader);
  OutputSection *ActualFirst = nullptr;
  for (OutputSection *Sec : OutputSections) {
    if (Sec->FirstInPtLoad == Out::ElfHeader) {
      ActualFirst = Sec;
      break;
    }
  }
  if (ActualFirst) {
    for (OutputSection *Sec : OutputSections)
      if (Sec->FirstInPtLoad == Out::ElfHeader)
        Sec->FirstInPtLoad = ActualFirst;
    FirstPTLoad->First = ActualFirst;
  } else {
    Phdrs.erase(FirstPTLoad);
  }

  auto PhdrI = std::find_if(Phdrs.begin(), Phdrs.end(), [](const PhdrEntry &E) {
    return E.p_type == PT_PHDR;
  });
  if (PhdrI != Phdrs.end())
    Phdrs.erase(PhdrI);
  return false;
}

void LinkerScript::assignAddresses(std::vector<PhdrEntry> &Phdrs) {
  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = 0;
  ErrorOnMissingSection = true;
  switchTo(Aether);

  for (BaseCommand *Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base)) {
      assignSymbol(Cmd, false);
      continue;
    }

    if (auto *Cmd = dyn_cast<AssertCommand>(Base)) {
      Cmd->Expression();
      continue;
    }

    auto *Cmd = cast<OutputSectionCommand>(Base);
    assignOffsets(Cmd);
  }

  uint64_t MinVA = std::numeric_limits<uint64_t>::max();
  for (OutputSection *Sec : *OutputSections) {
    if (Sec->Flags & SHF_ALLOC)
      MinVA = std::min<uint64_t>(MinVA, Sec->Addr);
    else
      Sec->Addr = 0;
  }

  allocateHeaders(Phdrs, *OutputSections, MinVA);
}

// Creates program headers as instructed by PHDRS linker script command.
std::vector<PhdrEntry> LinkerScript::createPhdrs() {
  std::vector<PhdrEntry> Ret;

  // Process PHDRS and FILEHDR keywords because they are not
  // real output sections and cannot be added in the following loop.
  for (const PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    Ret.emplace_back(Cmd.Type, Cmd.Flags == UINT_MAX ? PF_R : Cmd.Flags);
    PhdrEntry &Phdr = Ret.back();

    if (Cmd.HasFilehdr)
      Phdr.add(Out::ElfHeader);
    if (Cmd.HasPhdrs)
      Phdr.add(Out::ProgramHeaders);

    if (Cmd.LMAExpr) {
      Phdr.p_paddr = Cmd.LMAExpr().getValue();
      Phdr.HasLMA = true;
    }
  }

  // Add output sections to program headers.
  for (OutputSection *Sec : *OutputSections) {
    if (!(Sec->Flags & SHF_ALLOC))
      break;

    // Assign headers specified by linker script
    for (size_t Id : getPhdrIndices(Sec)) {
      Ret[Id].add(Sec);
      if (Opt.PhdrsCommands[Id].Flags == UINT_MAX)
        Ret[Id].p_flags |= Sec->getPhdrFlags();
    }
  }
  return Ret;
}

bool LinkerScript::ignoreInterpSection() {
  // Ignore .interp section in case we have PHDRS specification
  // and PT_INTERP isn't listed.
  if (Opt.PhdrsCommands.empty())
    return false;
  for (PhdrsCommand &Cmd : Opt.PhdrsCommands)
    if (Cmd.Type == PT_INTERP)
      return false;
  return true;
}

OutputSectionCommand *LinkerScript::getCmd(OutputSection *Sec) const {
  auto I = SecToCommand.find(Sec);
  if (I == SecToCommand.end())
    return nullptr;
  return I->second;
}

Optional<uint32_t> LinkerScript::getFiller(OutputSection *Sec) {
  if (OutputSectionCommand *Cmd = getCmd(Sec))
    return Cmd->Filler;
  return None;
}

static void writeInt(uint8_t *Buf, uint64_t Data, uint64_t Size) {
  if (Size == 1)
    *Buf = Data;
  else if (Size == 2)
    write16(Buf, Data, Config->Endianness);
  else if (Size == 4)
    write32(Buf, Data, Config->Endianness);
  else if (Size == 8)
    write64(Buf, Data, Config->Endianness);
  else
    llvm_unreachable("unsupported Size argument");
}

void LinkerScript::writeDataBytes(OutputSection *Sec, uint8_t *Buf) {
  if (OutputSectionCommand *Cmd = getCmd(Sec))
    for (BaseCommand *Base : Cmd->Commands)
      if (auto *Data = dyn_cast<BytesDataCommand>(Base))
        writeInt(Buf + Data->Offset, Data->Expression().getValue(), Data->Size);
}

bool LinkerScript::hasLMA(OutputSection *Sec) {
  if (OutputSectionCommand *Cmd = getCmd(Sec))
    if (Cmd->LMAExpr)
      return true;
  return false;
}

ExprValue LinkerScript::getSymbolValue(const Twine &Loc, StringRef S) {
  if (S == ".")
    return {CurOutSec, Dot - CurOutSec->Addr};
  if (SymbolBody *B = findSymbol(S)) {
    if (auto *D = dyn_cast<DefinedRegular>(B))
      return {D->Section, D->Value};
    if (auto *C = dyn_cast<DefinedCommon>(B))
      return {InX::Common, C->Offset};
  }
  error(Loc + ": symbol not found: " + S);
  return 0;
}

bool LinkerScript::isDefined(StringRef S) { return findSymbol(S) != nullptr; }

// Returns indices of ELF headers containing specific section. Each index is a
// zero based number of ELF header listed within PHDRS {} script block.
std::vector<size_t> LinkerScript::getPhdrIndices(OutputSection *Sec) {
  if (OutputSectionCommand *Cmd = getCmd(Sec)) {
    std::vector<size_t> Ret;
    for (StringRef PhdrName : Cmd->Phdrs)
      Ret.push_back(getPhdrIndex(Cmd->Location, PhdrName));
    return Ret;
  }
  return {};
}

size_t LinkerScript::getPhdrIndex(const Twine &Loc, StringRef PhdrName) {
  size_t I = 0;
  for (PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    if (Cmd.Name == PhdrName)
      return I;
    ++I;
  }
  error(Loc + ": section header '" + PhdrName + "' is not listed in PHDRS");
  return 0;
}
