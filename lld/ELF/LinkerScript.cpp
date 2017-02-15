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
#include "Driver.h"
#include "InputSection.h"
#include "Memory.h"
#include "OutputSections.h"
#include "ScriptLexer.h"
#include "Strings.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::elf;

LinkerScriptBase *elf::ScriptBase;
ScriptConfiguration *elf::ScriptConfig;

template <class ELFT> static SymbolBody *addRegular(SymbolAssignment *Cmd) {
  uint8_t Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  Symbol *Sym = Symtab<ELFT>::X->addUndefined(
      Cmd->Name, /*IsLocal=*/false, STB_GLOBAL, Visibility,
      /*Type*/ 0,
      /*CanOmitFromDynSym*/ false, /*File*/ nullptr);

  replaceBody<DefinedRegular<ELFT>>(Sym, Cmd->Name, /*IsLocal=*/false,
                                    Visibility, STT_NOTYPE, 0, 0, nullptr,
                                    nullptr);
  return Sym->body();
}

template <class ELFT> static SymbolBody *addSynthetic(SymbolAssignment *Cmd) {
  uint8_t Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  const OutputSectionBase *Sec =
      ScriptConfig->HasSections ? nullptr : Cmd->Expression.Section();
  Symbol *Sym = Symtab<ELFT>::X->addUndefined(
      Cmd->Name, /*IsLocal=*/false, STB_GLOBAL, Visibility,
      /*Type*/ 0,
      /*CanOmitFromDynSym*/ false, /*File*/ nullptr);

  replaceBody<DefinedSynthetic>(Sym, Cmd->Name, 0, Sec);
  return Sym->body();
}

static bool isUnderSysroot(StringRef Path) {
  if (Config->Sysroot == "")
    return false;
  for (; !Path.empty(); Path = sys::path::parent_path(Path))
    if (sys::fs::equivalent(Config->Sysroot, Path))
      return true;
  return false;
}

// Sets value of a symbol. Two kinds of symbols are processed: synthetic
// symbols, whose value is an offset from beginning of section and regular
// symbols whose value is absolute.
template <class ELFT>
static void assignSymbol(SymbolAssignment *Cmd, typename ELFT::uint Dot = 0) {
  if (!Cmd->Sym)
    return;

  if (auto *Body = dyn_cast<DefinedSynthetic>(Cmd->Sym)) {
    Body->Section = Cmd->Expression.Section();
    if (Body->Section) {
      uint64_t VA = 0;
      if (Body->Section->Flags & SHF_ALLOC)
        VA = Body->Section->Addr;
      Body->Value = Cmd->Expression(Dot) - VA;
    }
    return;
  }

  cast<DefinedRegular<ELFT>>(Cmd->Sym)->Value = Cmd->Expression(Dot);
}

template <class ELFT> static void addSymbol(SymbolAssignment *Cmd) {
  if (Cmd->Name == ".")
    return;

  // If a symbol was in PROVIDE(), we need to define it only when
  // it is a referenced undefined symbol.
  SymbolBody *B = Symtab<ELFT>::X->find(Cmd->Name);
  if (Cmd->Provide && (!B || B->isDefined()))
    return;

  // Otherwise, create a new symbol if one does not exist or an
  // undefined one does exist.
  if (Cmd->Expression.IsAbsolute())
    Cmd->Sym = addRegular<ELFT>(Cmd);
  else
    Cmd->Sym = addSynthetic<ELFT>(Cmd);

  // If there are sections, then let the value be assigned later in
  // `assignAddresses`.
  if (!ScriptConfig->HasSections)
    assignSymbol<ELFT>(Cmd);
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

template <class ELFT> LinkerScript<ELFT>::LinkerScript() = default;
template <class ELFT> LinkerScript<ELFT>::~LinkerScript() = default;

template <class ELFT> static StringRef basename(InputSectionBase<ELFT> *S) {
  if (S->getFile())
    return sys::path::filename(S->getFile()->getName());
  return "";
}

template <class ELFT>
bool LinkerScript<ELFT>::shouldKeep(InputSectionBase<ELFT> *S) {
  for (InputSectionDescription *ID : Opt.KeptSections)
    if (ID->FilePat.match(basename(S)))
      for (SectionPattern &P : ID->SectionPatterns)
        if (P.SectionPat.match(S->Name))
          return true;
  return false;
}

static bool comparePriority(InputSectionData *A, InputSectionData *B) {
  return getPriority(A->Name) < getPriority(B->Name);
}

static bool compareName(InputSectionData *A, InputSectionData *B) {
  return A->Name < B->Name;
}

static bool compareAlignment(InputSectionData *A, InputSectionData *B) {
  // ">" is not a mistake. Larger alignments are placed before smaller
  // alignments in order to reduce the amount of padding necessary.
  // This is compatible with GNU.
  return A->Alignment > B->Alignment;
}

static std::function<bool(InputSectionData *, InputSectionData *)>
getComparator(SortSectionPolicy K) {
  switch (K) {
  case SortSectionPolicy::Alignment:
    return compareAlignment;
  case SortSectionPolicy::Name:
    return compareName;
  case SortSectionPolicy::Priority:
    return comparePriority;
  default:
    llvm_unreachable("unknown sort policy");
  }
}

template <class ELFT>
static bool matchConstraints(ArrayRef<InputSectionBase<ELFT> *> Sections,
                             ConstraintKind Kind) {
  if (Kind == ConstraintKind::NoConstraint)
    return true;
  bool IsRW = llvm::any_of(Sections, [=](InputSectionData *Sec2) {
    auto *Sec = static_cast<InputSectionBase<ELFT> *>(Sec2);
    return Sec->Flags & SHF_WRITE;
  });
  return (IsRW && Kind == ConstraintKind::ReadWrite) ||
         (!IsRW && Kind == ConstraintKind::ReadOnly);
}

static void sortSections(InputSectionData **Begin, InputSectionData **End,
                         SortSectionPolicy K) {
  if (K != SortSectionPolicy::Default && K != SortSectionPolicy::None)
    std::stable_sort(Begin, End, getComparator(K));
}

// Compute and remember which sections the InputSectionDescription matches.
template <class ELFT>
void LinkerScript<ELFT>::computeInputSections(InputSectionDescription *I) {
  // Collects all sections that satisfy constraints of I
  // and attach them to I.
  for (SectionPattern &Pat : I->SectionPatterns) {
    size_t SizeBefore = I->Sections.size();

    for (InputSectionBase<ELFT> *S : Symtab<ELFT>::X->Sections) {
      if (!S->Live || S->Assigned)
        continue;

      StringRef Filename = basename(S);
      if (!I->FilePat.match(Filename) || Pat.ExcludedFilePat.match(Filename))
        continue;
      if (!Pat.SectionPat.match(S->Name))
        continue;
      I->Sections.push_back(S);
      S->Assigned = true;
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
    InputSectionData **Begin = I->Sections.data() + SizeBefore;
    InputSectionData **End = I->Sections.data() + I->Sections.size();
    if (Pat.SortOuter != SortSectionPolicy::None) {
      if (Pat.SortInner == SortSectionPolicy::Default)
        sortSections(Begin, End, Config->SortSection);
      else
        sortSections(Begin, End, Pat.SortInner);
      sortSections(Begin, End, Pat.SortOuter);
    }
  }
}

template <class ELFT>
void LinkerScript<ELFT>::discard(ArrayRef<InputSectionBase<ELFT> *> V) {
  for (InputSectionBase<ELFT> *S : V) {
    S->Live = false;
    reportDiscarded(S);
  }
}

template <class ELFT>
std::vector<InputSectionBase<ELFT> *>
LinkerScript<ELFT>::createInputSectionList(OutputSectionCommand &OutCmd) {
  std::vector<InputSectionBase<ELFT> *> Ret;

  for (const std::unique_ptr<BaseCommand> &Base : OutCmd.Commands) {
    auto *Cmd = dyn_cast<InputSectionDescription>(Base.get());
    if (!Cmd)
      continue;
    computeInputSections(Cmd);
    for (InputSectionData *S : Cmd->Sections)
      Ret.push_back(static_cast<InputSectionBase<ELFT> *>(S));
  }

  return Ret;
}

template <class ELFT>
void LinkerScript<ELFT>::addSection(OutputSectionFactory<ELFT> &Factory,
                                    InputSectionBase<ELFT> *Sec,
                                    StringRef Name) {
  OutputSectionBase *OutSec;
  bool IsNew;
  std::tie(OutSec, IsNew) = Factory.create(Sec, Name);
  if (IsNew)
    OutputSections->push_back(OutSec);
  OutSec->addSection(Sec);
}

template <class ELFT>
void LinkerScript<ELFT>::processCommands(OutputSectionFactory<ELFT> &Factory) {
  for (unsigned I = 0; I < Opt.Commands.size(); ++I) {
    auto Iter = Opt.Commands.begin() + I;
    const std::unique_ptr<BaseCommand> &Base1 = *Iter;

    // Handle symbol assignments outside of any output section.
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base1.get())) {
      addSymbol<ELFT>(Cmd);
      continue;
    }

    if (auto *Cmd = dyn_cast<AssertCommand>(Base1.get())) {
      // If we don't have SECTIONS then output sections have already been
      // created by Writer<ELFT>. The LinkerScript<ELFT>::assignAddresses
      // will not be called, so ASSERT should be evaluated now.
      if (!Opt.HasSections)
        Cmd->Expression(0);
      continue;
    }

    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base1.get())) {
      std::vector<InputSectionBase<ELFT> *> V = createInputSectionList(*Cmd);

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
      if (!matchConstraints<ELFT>(V, Cmd->Constraint)) {
        for (InputSectionBase<ELFT> *S : V)
          S->Assigned = false;
        Opt.Commands.erase(Iter);
        --I;
        continue;
      }

      // A directive may contain symbol definitions like this:
      // ".foo : { ...; bar = .; }". Handle them.
      for (const std::unique_ptr<BaseCommand> &Base : Cmd->Commands)
        if (auto *OutCmd = dyn_cast<SymbolAssignment>(Base.get()))
          addSymbol<ELFT>(OutCmd);

      // Handle subalign (e.g. ".foo : SUBALIGN(32) { ... }"). If subalign
      // is given, input sections are aligned to that value, whether the
      // given value is larger or smaller than the original section alignment.
      if (Cmd->SubalignExpr) {
        uint32_t Subalign = Cmd->SubalignExpr(0);
        for (InputSectionBase<ELFT> *S : V)
          S->Alignment = Subalign;
      }

      // Add input sections to an output section.
      for (InputSectionBase<ELFT> *S : V)
        addSection(Factory, S, Cmd->Name);
    }
  }
}

// Add sections that didn't match any sections command.
template <class ELFT>
void LinkerScript<ELFT>::addOrphanSections(
    OutputSectionFactory<ELFT> &Factory) {
  for (InputSectionBase<ELFT> *S : Symtab<ELFT>::X->Sections)
    if (S->Live && !S->OutSec)
      addSection(Factory, S, getOutputSectionName(S->Name));
}

template <class ELFT> static bool isTbss(OutputSectionBase *Sec) {
  return (Sec->Flags & SHF_TLS) && Sec->Type == SHT_NOBITS;
}

template <class ELFT> void LinkerScript<ELFT>::output(InputSection<ELFT> *S) {
  if (!AlreadyOutputIS.insert(S).second)
    return;
  bool IsTbss = isTbss<ELFT>(CurOutSec);

  uintX_t Pos = IsTbss ? Dot + ThreadBssOffset : Dot;
  Pos = alignTo(Pos, S->Alignment);
  S->OutSecOff = Pos - CurOutSec->Addr;
  Pos += S->getSize();

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

  if (IsTbss)
    ThreadBssOffset = Pos - Dot;
  else
    Dot = Pos;
}

template <class ELFT> void LinkerScript<ELFT>::flush() {
  if (!CurOutSec || !AlreadyOutputOS.insert(CurOutSec).second)
    return;
  if (auto *OutSec = dyn_cast<OutputSection<ELFT>>(CurOutSec)) {
    for (InputSection<ELFT> *I : OutSec->Sections)
      output(I);
  } else {
    Dot += CurOutSec->Size;
  }
}

template <class ELFT>
void LinkerScript<ELFT>::switchTo(OutputSectionBase *Sec) {
  if (CurOutSec == Sec)
    return;
  if (AlreadyOutputOS.count(Sec))
    return;

  flush();
  CurOutSec = Sec;

  Dot = alignTo(Dot, CurOutSec->Addralign);
  CurOutSec->Addr = isTbss<ELFT>(CurOutSec) ? Dot + ThreadBssOffset : Dot;

  // If neither AT nor AT> is specified for an allocatable section, the linker
  // will set the LMA such that the difference between VMA and LMA for the
  // section is the same as the preceding output section in the same region
  // https://sourceware.org/binutils/docs-2.20/ld/Output-Section-LMA.html
  CurOutSec->setLMAOffset(LMAOffset);
}

template <class ELFT> void LinkerScript<ELFT>::process(BaseCommand &Base) {
  // This handles the assignments to symbol or to a location counter (.)
  if (auto *AssignCmd = dyn_cast<SymbolAssignment>(&Base)) {
    if (AssignCmd->Name == ".") {
      // Update to location counter means update to section size.
      uintX_t Val = AssignCmd->Expression(Dot);
      if (Val < Dot)
        error("unable to move location counter backward for: " +
              CurOutSec->Name);
      Dot = Val;
      CurOutSec->Size = Dot - CurOutSec->Addr;
      return;
    }
    assignSymbol<ELFT>(AssignCmd, Dot);
    return;
  }

  // Handle BYTE(), SHORT(), LONG(), or QUAD().
  if (auto *DataCmd = dyn_cast<BytesDataCommand>(&Base)) {
    DataCmd->Offset = Dot - CurOutSec->Addr;
    Dot += DataCmd->Size;
    CurOutSec->Size = Dot - CurOutSec->Addr;
    return;
  }

  if (auto *AssertCmd = dyn_cast<AssertCommand>(&Base)) {
    AssertCmd->Expression(Dot);
    return;
  }

  // It handles single input section description command,
  // calculates and assigns the offsets for each section and also
  // updates the output section size.
  auto &ICmd = cast<InputSectionDescription>(Base);
  for (InputSectionData *ID : ICmd.Sections) {
    // We tentatively added all synthetic sections at the beginning and removed
    // empty ones afterwards (because there is no way to know whether they were
    // going be empty or not other than actually running linker scripts.)
    // We need to ignore remains of empty sections.
    if (auto *Sec = dyn_cast<SyntheticSection<ELFT>>(ID))
      if (Sec->empty())
        continue;

    auto *IB = static_cast<InputSectionBase<ELFT> *>(ID);
    switchTo(IB->OutSec);
    if (auto *I = dyn_cast<InputSection<ELFT>>(IB))
      output(I);
    else
      flush();
  }
}

template <class ELFT>
static OutputSectionBase *
findSection(StringRef Name, const std::vector<OutputSectionBase *> &Sections) {
  auto End = Sections.end();
  auto HasName = [=](OutputSectionBase *Sec) { return Sec->getName() == Name; };
  auto I = std::find_if(Sections.begin(), End, HasName);
  std::vector<OutputSectionBase *> Ret;
  if (I == End)
    return nullptr;
  assert(std::find_if(I + 1, End, HasName) == End);
  return *I;
}

// This function searches for a memory region to place the given output
// section in. If found, a pointer to the appropriate memory region is
// returned. Otherwise, a nullptr is returned.
template <class ELFT>
MemoryRegion *LinkerScript<ELFT>::findMemoryRegion(OutputSectionCommand *Cmd,
                                                   OutputSectionBase *Sec) {
  // If a memory region name was specified in the output section command,
  // then try to find that region first.
  if (!Cmd->MemoryRegionName.empty()) {
    auto It = Opt.MemoryRegions.find(Cmd->MemoryRegionName);
    if (It != Opt.MemoryRegions.end())
      return &It->second;
    error("memory region '" + Cmd->MemoryRegionName + "' not declared");
    return nullptr;
  }

  // The memory region name is empty, thus a suitable region must be
  // searched for in the region map. If the region map is empty, just
  // return. Note that this check doesn't happen at the very beginning
  // so that uses of undeclared regions can be caught.
  if (!Opt.MemoryRegions.size())
    return nullptr;

  // See if a region can be found by matching section flags.
  for (auto &MRI : Opt.MemoryRegions) {
    MemoryRegion &MR = MRI.second;
    if ((MR.Flags & Sec->Flags) != 0 && (MR.NegFlags & Sec->Flags) == 0)
      return &MR;
  }

  // Otherwise, no suitable region was found.
  if (Sec->Flags & SHF_ALLOC)
    error("no memory region specified for section '" + Sec->Name + "'");
  return nullptr;
}

// This function assigns offsets to input sections and an output section
// for a single sections command (e.g. ".text { *(.text); }").
template <class ELFT>
void LinkerScript<ELFT>::assignOffsets(OutputSectionCommand *Cmd) {
  if (Cmd->LMAExpr)
    LMAOffset = Cmd->LMAExpr(Dot) - Dot;
  OutputSectionBase *Sec = findSection<ELFT>(Cmd->Name, *OutputSections);
  if (!Sec)
    return;

  // Handle align (e.g. ".foo : ALIGN(16) { ... }").
  if (Cmd->AlignExpr)
    Sec->updateAlignment(Cmd->AlignExpr(0));

  // Try and find an appropriate memory region to assign offsets in.
  CurMemRegion = findMemoryRegion(Cmd, Sec);
  if (CurMemRegion)
    Dot = CurMemRegion->Offset;
  switchTo(Sec);

  // Find the last section output location. We will output orphan sections
  // there so that end symbols point to the correct location.
  auto E = std::find_if(Cmd->Commands.rbegin(), Cmd->Commands.rend(),
                        [](const std::unique_ptr<BaseCommand> &Cmd) {
                          return !isa<SymbolAssignment>(*Cmd);
                        })
               .base();
  for (auto I = Cmd->Commands.begin(); I != E; ++I)
    process(**I);
  flush();
  std::for_each(E, Cmd->Commands.end(),
                [this](std::unique_ptr<BaseCommand> &B) { process(*B.get()); });
}

template <class ELFT> void LinkerScript<ELFT>::removeEmptyCommands() {
  // It is common practice to use very generic linker scripts. So for any
  // given run some of the output sections in the script will be empty.
  // We could create corresponding empty output sections, but that would
  // clutter the output.
  // We instead remove trivially empty sections. The bfd linker seems even
  // more aggressive at removing them.
  auto Pos = std::remove_if(
      Opt.Commands.begin(), Opt.Commands.end(),
      [&](const std::unique_ptr<BaseCommand> &Base) {
        if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
          return !findSection<ELFT>(Cmd->Name, *OutputSections);
        return false;
      });
  Opt.Commands.erase(Pos, Opt.Commands.end());
}

static bool isAllSectionDescription(const OutputSectionCommand &Cmd) {
  for (const std::unique_ptr<BaseCommand> &I : Cmd.Commands)
    if (!isa<InputSectionDescription>(*I))
      return false;
  return true;
}

template <class ELFT> void LinkerScript<ELFT>::adjustSectionsBeforeSorting() {
  // If the output section contains only symbol assignments, create a
  // corresponding output section. The bfd linker seems to only create them if
  // '.' is assigned to, but creating these section should not have any bad
  // consequeces and gives us a section to put the symbol in.
  uintX_t Flags = SHF_ALLOC;
  uint32_t Type = SHT_NOBITS;
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
    if (!Cmd)
      continue;
    if (OutputSectionBase *Sec =
            findSection<ELFT>(Cmd->Name, *OutputSections)) {
      Flags = Sec->Flags;
      Type = Sec->Type;
      continue;
    }

    if (isAllSectionDescription(*Cmd))
      continue;

    auto *OutSec = make<OutputSection<ELFT>>(Cmd->Name, Type, Flags);
    OutputSections->push_back(OutSec);
  }
}

template <class ELFT> void LinkerScript<ELFT>::adjustSectionsAfterSorting() {
  placeOrphanSections();

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
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
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
static bool shouldSkip(const BaseCommand &Cmd) {
  if (isa<OutputSectionCommand>(Cmd))
    return false;
  const auto *Assign = dyn_cast<SymbolAssignment>(&Cmd);
  if (!Assign)
    return true;
  return Assign->Name != ".";
}

// Orphan sections are sections present in the input files which are
// not explicitly placed into the output file by the linker script.
//
// When the control reaches this function, Opt.Commands contains
// output section commands for non-orphan sections only. This function
// adds new elements for orphan sections to Opt.Commands so that all
// sections are explicitly handled by Opt.Commands.
//
// Writer<ELFT>::sortSections has already sorted output sections.
// What we need to do is to scan OutputSections vector and
// Opt.Commands in parallel to find orphan sections. If there is an
// output section that doesn't have a corresponding entry in
// Opt.Commands, we will insert a new entry to Opt.Commands.
//
// There is some ambiguity as to where exactly a new entry should be
// inserted, because Opt.Commands contains not only output section
// commands but other types of commands such as symbol assignment
// expressions. There's no correct answer here due to the lack of the
// formal specification of the linker script. We use heuristics to
// determine whether a new output command should be added before or
// after another commands. For the details, look at shouldSkip
// function.
template <class ELFT> void LinkerScript<ELFT>::placeOrphanSections() {
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
                   [](const std::unique_ptr<BaseCommand> &Cmd) {
                     if (isa<OutputSectionCommand>(*Cmd))
                       return true;
                     const auto *Assign = dyn_cast<SymbolAssignment>(Cmd.get());
                     if (!Assign)
                       return false;
                     return Assign->Name == ".";
                   });
  if (FirstSectionOrDotAssignment != Opt.Commands.end()) {
    CmdIndex = FirstSectionOrDotAssignment - Opt.Commands.begin();
    if (isa<SymbolAssignment>(**FirstSectionOrDotAssignment))
      ++CmdIndex;
  }

  for (OutputSectionBase *Sec : *OutputSections) {
    StringRef Name = Sec->getName();

    // Find the last spot where we can insert a command and still get the
    // correct result.
    auto CmdIter = Opt.Commands.begin() + CmdIndex;
    auto E = Opt.Commands.end();
    while (CmdIter != E && shouldSkip(**CmdIter)) {
      ++CmdIter;
      ++CmdIndex;
    }

    auto Pos =
        std::find_if(CmdIter, E, [&](const std::unique_ptr<BaseCommand> &Base) {
          auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
          return Cmd && Cmd->Name == Name;
        });
    if (Pos == E) {
      Opt.Commands.insert(CmdIter,
                          llvm::make_unique<OutputSectionCommand>(Name));
      ++CmdIndex;
      continue;
    }

    // Continue from where we found it.
    CmdIndex = (Pos - Opt.Commands.begin()) + 1;
  }
}

template <class ELFT>
void LinkerScript<ELFT>::assignAddresses(std::vector<PhdrEntry> &Phdrs) {
  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = 0;

  // A symbol can be assigned before any section is mentioned in the linker
  // script. In an DSO, the symbol values are addresses, so the only important
  // section values are:
  // * SHN_UNDEF
  // * SHN_ABS
  // * Any value meaning a regular section.
  // To handle that, create a dummy aether section that fills the void before
  // the linker scripts switches to another section. It has an index of one
  // which will map to whatever the first actual section is.
  auto *Aether = make<OutputSectionBase>("", 0, SHF_ALLOC);
  Aether->SectionIndex = 1;
  switchTo(Aether);

  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base.get())) {
      if (Cmd->Name == ".") {
        Dot = Cmd->Expression(Dot);
      } else if (Cmd->Sym) {
        assignSymbol<ELFT>(Cmd, Dot);
      }
      continue;
    }

    if (auto *Cmd = dyn_cast<AssertCommand>(Base.get())) {
      Cmd->Expression(Dot);
      continue;
    }

    auto *Cmd = cast<OutputSectionCommand>(Base.get());
    if (Cmd->AddrExpr)
      Dot = Cmd->AddrExpr(Dot);
    assignOffsets(Cmd);
  }

  uintX_t MinVA = std::numeric_limits<uintX_t>::max();
  for (OutputSectionBase *Sec : *OutputSections) {
    if (Sec->Flags & SHF_ALLOC)
      MinVA = std::min<uint64_t>(MinVA, Sec->Addr);
    else
      Sec->Addr = 0;
  }

  allocateHeaders<ELFT>(Phdrs, *OutputSections, MinVA);
}

// Creates program headers as instructed by PHDRS linker script command.
template <class ELFT> std::vector<PhdrEntry> LinkerScript<ELFT>::createPhdrs() {
  std::vector<PhdrEntry> Ret;

  // Process PHDRS and FILEHDR keywords because they are not
  // real output sections and cannot be added in the following loop.
  for (const PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    Ret.emplace_back(Cmd.Type, Cmd.Flags == UINT_MAX ? PF_R : Cmd.Flags);
    PhdrEntry &Phdr = Ret.back();

    if (Cmd.HasFilehdr)
      Phdr.add(Out<ELFT>::ElfHeader);
    if (Cmd.HasPhdrs)
      Phdr.add(Out<ELFT>::ProgramHeaders);

    if (Cmd.LMAExpr) {
      Phdr.p_paddr = Cmd.LMAExpr(0);
      Phdr.HasLMA = true;
    }
  }

  // Add output sections to program headers.
  for (OutputSectionBase *Sec : *OutputSections) {
    if (!(Sec->Flags & SHF_ALLOC))
      break;

    // Assign headers specified by linker script
    for (size_t Id : getPhdrIndices(Sec->getName())) {
      Ret[Id].add(Sec);
      if (Opt.PhdrsCommands[Id].Flags == UINT_MAX)
        Ret[Id].p_flags |= Sec->getPhdrFlags();
    }
  }
  return Ret;
}

template <class ELFT> bool LinkerScript<ELFT>::ignoreInterpSection() {
  // Ignore .interp section in case we have PHDRS specification
  // and PT_INTERP isn't listed.
  return !Opt.PhdrsCommands.empty() &&
         llvm::find_if(Opt.PhdrsCommands, [](const PhdrsCommand &Cmd) {
           return Cmd.Type == PT_INTERP;
         }) == Opt.PhdrsCommands.end();
}

template <class ELFT> uint32_t LinkerScript<ELFT>::getFiller(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->Name == Name)
        return Cmd->Filler;
  return 0;
}

template <class ELFT>
static void writeInt(uint8_t *Buf, uint64_t Data, uint64_t Size) {
  const endianness E = ELFT::TargetEndianness;

  switch (Size) {
  case 1:
    *Buf = (uint8_t)Data;
    break;
  case 2:
    write16<E>(Buf, Data);
    break;
  case 4:
    write32<E>(Buf, Data);
    break;
  case 8:
    write64<E>(Buf, Data);
    break;
  default:
    llvm_unreachable("unsupported Size argument");
  }
}

template <class ELFT>
void LinkerScript<ELFT>::writeDataBytes(StringRef Name, uint8_t *Buf) {
  int I = getSectionIndex(Name);
  if (I == INT_MAX)
    return;

  auto *Cmd = dyn_cast<OutputSectionCommand>(Opt.Commands[I].get());
  for (const std::unique_ptr<BaseCommand> &Base : Cmd->Commands)
    if (auto *Data = dyn_cast<BytesDataCommand>(Base.get()))
      writeInt<ELFT>(Buf + Data->Offset, Data->Expression(0), Data->Size);
}

template <class ELFT> bool LinkerScript<ELFT>::hasLMA(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->LMAExpr && Cmd->Name == Name)
        return true;
  return false;
}

// Returns the index of the given section name in linker script
// SECTIONS commands. Sections are laid out as the same order as they
// were in the script. If a given name did not appear in the script,
// it returns INT_MAX, so that it will be laid out at end of file.
template <class ELFT> int LinkerScript<ELFT>::getSectionIndex(StringRef Name) {
  for (int I = 0, E = Opt.Commands.size(); I != E; ++I)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Opt.Commands[I].get()))
      if (Cmd->Name == Name)
        return I;
  return INT_MAX;
}

template <class ELFT> bool LinkerScript<ELFT>::hasPhdrsCommands() {
  return !Opt.PhdrsCommands.empty();
}

template <class ELFT>
const OutputSectionBase *LinkerScript<ELFT>::getOutputSection(const Twine &Loc,
                                                              StringRef Name) {
  static OutputSectionBase FakeSec("", 0, 0);

  for (OutputSectionBase *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec;

  error(Loc + ": undefined section " + Name);
  return &FakeSec;
}

// This function is essentially the same as getOutputSection(Name)->Size,
// but it won't print out an error message if a given section is not found.
//
// Linker script does not create an output section if its content is empty.
// We want to allow SIZEOF(.foo) where .foo is a section which happened to
// be empty. That is why this function is different from getOutputSection().
template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionSize(StringRef Name) {
  for (OutputSectionBase *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->Size;
  return 0;
}

template <class ELFT> uint64_t LinkerScript<ELFT>::getHeaderSize() {
  return elf::getHeaderSize<ELFT>();
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getSymbolValue(const Twine &Loc, StringRef S) {
  if (SymbolBody *B = Symtab<ELFT>::X->find(S))
    return B->getVA<ELFT>();
  error(Loc + ": symbol not found: " + S);
  return 0;
}

template <class ELFT> bool LinkerScript<ELFT>::isDefined(StringRef S) {
  return Symtab<ELFT>::X->find(S) != nullptr;
}

template <class ELFT> bool LinkerScript<ELFT>::isAbsolute(StringRef S) {
  SymbolBody *Sym = Symtab<ELFT>::X->find(S);
  auto *DR = dyn_cast_or_null<DefinedRegular<ELFT>>(Sym);
  return DR && !DR->Section;
}

// Gets section symbol belongs to. Symbol "." doesn't belong to any
// specific section but isn't absolute at the same time, so we try
// to find suitable section for it as well.
template <class ELFT>
const OutputSectionBase *LinkerScript<ELFT>::getSymbolSection(StringRef S) {
  if (SymbolBody *Sym = Symtab<ELFT>::X->find(S))
    return SymbolTableSection<ELFT>::getOutputSection(Sym);
  return CurOutSec;
}

// Returns indices of ELF headers containing specific section, identified
// by Name. Each index is a zero based number of ELF header listed within
// PHDRS {} script block.
template <class ELFT>
std::vector<size_t> LinkerScript<ELFT>::getPhdrIndices(StringRef SectionName) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
    if (!Cmd || Cmd->Name != SectionName)
      continue;

    std::vector<size_t> Ret;
    for (StringRef PhdrName : Cmd->Phdrs)
      Ret.push_back(getPhdrIndex(Cmd->Location, PhdrName));
    return Ret;
  }
  return {};
}

template <class ELFT>
size_t LinkerScript<ELFT>::getPhdrIndex(const Twine &Loc, StringRef PhdrName) {
  size_t I = 0;
  for (PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    if (Cmd.Name == PhdrName)
      return I;
    ++I;
  }
  error(Loc + ": section header '" + PhdrName + "' is not listed in PHDRS");
  return 0;
}

class elf::ScriptParser final : public ScriptLexer {
  typedef void (ScriptParser::*Handler)();

public:
  ScriptParser(MemoryBufferRef MB)
      : ScriptLexer(MB),
        IsUnderSysroot(isUnderSysroot(MB.getBufferIdentifier())) {}

  void readLinkerScript();
  void readVersionScript();
  void readDynamicList();

private:
  void addFile(StringRef Path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readMemory();
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readPhdrs();
  void readSearchDir();
  void readSections();
  void readVersion();
  void readVersionScriptCommand();

  SymbolAssignment *readAssignment(StringRef Name);
  BytesDataCommand *readBytesDataCommand(StringRef Tok);
  uint32_t readFill();
  OutputSectionCommand *readOutputSectionDescription(StringRef OutSec);
  uint32_t readOutputSectionFiller(StringRef Tok);
  std::vector<StringRef> readOutputSectionPhdrs();
  InputSectionDescription *readInputSectionDescription(StringRef Tok);
  StringMatcher readFilePatterns();
  std::vector<SectionPattern> readInputSectionsList();
  InputSectionDescription *readInputSectionRules(StringRef FilePattern);
  unsigned readPhdrType();
  SortSectionPolicy readSortKind();
  SymbolAssignment *readProvideHidden(bool Provide, bool Hidden);
  SymbolAssignment *readProvideOrAssignment(StringRef Tok);
  void readSort();
  Expr readAssert();

  uint64_t readMemoryAssignment(StringRef, StringRef, StringRef);
  std::pair<uint32_t, uint32_t> readMemoryAttributes();

  Expr readExpr();
  Expr readExpr1(Expr Lhs, int MinPrec);
  StringRef readParenLiteral();
  Expr readPrimary();
  Expr readTernary(Expr Cond);
  Expr readParenExpr();

  // For parsing version script.
  std::vector<SymbolVersion> readVersionExtern();
  void readAnonymousDeclaration();
  void readVersionDeclaration(StringRef VerStr);
  std::vector<SymbolVersion> readSymbols();
  void readLocals();

  ScriptConfiguration &Opt = *ScriptConfig;
  bool IsUnderSysroot;
};

void ScriptParser::readDynamicList() {
  expect("{");
  readAnonymousDeclaration();
  if (!atEOF())
    setError("EOF expected, but got " + next());
}

void ScriptParser::readVersionScript() {
  readVersionScriptCommand();
  if (!atEOF())
    setError("EOF expected, but got " + next());
}

void ScriptParser::readVersionScriptCommand() {
  if (consume("{")) {
    readAnonymousDeclaration();
    return;
  }

  while (!atEOF() && !Error && peek() != "}") {
    StringRef VerStr = next();
    if (VerStr == "{") {
      setError("anonymous version definition is used in "
               "combination with other version definitions");
      return;
    }
    expect("{");
    readVersionDeclaration(VerStr);
  }
}

void ScriptParser::readVersion() {
  expect("{");
  readVersionScriptCommand();
  expect("}");
}

void ScriptParser::readLinkerScript() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Tok == ";")
      continue;

    if (Tok == "ASSERT") {
      Opt.Commands.emplace_back(new AssertCommand(readAssert()));
    } else if (Tok == "ENTRY") {
      readEntry();
    } else if (Tok == "EXTERN") {
      readExtern();
    } else if (Tok == "GROUP" || Tok == "INPUT") {
      readGroup();
    } else if (Tok == "INCLUDE") {
      readInclude();
    } else if (Tok == "MEMORY") {
      readMemory();
    } else if (Tok == "OUTPUT") {
      readOutput();
    } else if (Tok == "OUTPUT_ARCH") {
      readOutputArch();
    } else if (Tok == "OUTPUT_FORMAT") {
      readOutputFormat();
    } else if (Tok == "PHDRS") {
      readPhdrs();
    } else if (Tok == "SEARCH_DIR") {
      readSearchDir();
    } else if (Tok == "SECTIONS") {
      readSections();
    } else if (Tok == "VERSION") {
      readVersion();
    } else if (SymbolAssignment *Cmd = readProvideOrAssignment(Tok)) {
      Opt.Commands.emplace_back(Cmd);
    } else {
      setError("unknown directive: " + Tok);
    }
  }
}

void ScriptParser::addFile(StringRef S) {
  if (IsUnderSysroot && S.startswith("/")) {
    SmallString<128> PathData;
    StringRef Path = (Config->Sysroot + S).toStringRef(PathData);
    if (sys::fs::exists(Path)) {
      Driver->addFile(Saver.save(Path));
      return;
    }
  }

  if (sys::path::is_absolute(S)) {
    Driver->addFile(S);
  } else if (S.startswith("=")) {
    if (Config->Sysroot.empty())
      Driver->addFile(S.substr(1));
    else
      Driver->addFile(Saver.save(Config->Sysroot + "/" + S.substr(1)));
  } else if (S.startswith("-l")) {
    Driver->addLibrary(S.substr(2));
  } else if (sys::fs::exists(S)) {
    Driver->addFile(S);
  } else {
    if (Optional<std::string> Path = findFromSearchPaths(S))
      Driver->addFile(Saver.save(*Path));
    else
      setError("unable to find " + S);
  }
}

void ScriptParser::readAsNeeded() {
  expect("(");
  bool Orig = Config->AsNeeded;
  Config->AsNeeded = true;
  while (!Error && !consume(")"))
    addFile(unquote(next()));
  Config->AsNeeded = Orig;
}

void ScriptParser::readEntry() {
  // -e <symbol> takes predecence over ENTRY(<symbol>).
  expect("(");
  StringRef Tok = next();
  if (Config->Entry.empty())
    Config->Entry = Tok;
  expect(")");
}

void ScriptParser::readExtern() {
  expect("(");
  while (!Error && !consume(")"))
    Config->Undefined.push_back(next());
}

void ScriptParser::readGroup() {
  expect("(");
  while (!Error && !consume(")")) {
    StringRef Tok = next();
    if (Tok == "AS_NEEDED")
      readAsNeeded();
    else
      addFile(unquote(Tok));
  }
}

void ScriptParser::readInclude() {
  StringRef Tok = unquote(next());

  // https://sourceware.org/binutils/docs/ld/File-Commands.html:
  // The file will be searched for in the current directory, and in any
  // directory specified with the -L option.
  if (sys::fs::exists(Tok)) {
    if (Optional<MemoryBufferRef> MB = readFile(Tok))
      tokenize(*MB);
    return;
  }
  if (Optional<std::string> Path = findFromSearchPaths(Tok)) {
    if (Optional<MemoryBufferRef> MB = readFile(*Path))
      tokenize(*MB);
    return;
  }
  setError("cannot open " + Tok);
}

void ScriptParser::readOutput() {
  // -o <file> takes predecence over OUTPUT(<file>).
  expect("(");
  StringRef Tok = next();
  if (Config->OutputFile.empty())
    Config->OutputFile = unquote(Tok);
  expect(")");
}

void ScriptParser::readOutputArch() {
  // OUTPUT_ARCH is ignored for now.
  expect("(");
  while (!Error && !consume(")"))
    skip();
}

void ScriptParser::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  skip();
  StringRef Tok = next();
  if (Tok == ")")
    return;
  if (Tok != ",") {
    setError("unexpected token: " + Tok);
    return;
  }
  skip();
  expect(",");
  skip();
  expect(")");
}

void ScriptParser::readPhdrs() {
  expect("{");
  while (!Error && !consume("}")) {
    StringRef Tok = next();
    Opt.PhdrsCommands.push_back(
        {Tok, PT_NULL, false, false, UINT_MAX, nullptr});
    PhdrsCommand &PhdrCmd = Opt.PhdrsCommands.back();

    PhdrCmd.Type = readPhdrType();
    do {
      Tok = next();
      if (Tok == ";")
        break;
      if (Tok == "FILEHDR")
        PhdrCmd.HasFilehdr = true;
      else if (Tok == "PHDRS")
        PhdrCmd.HasPhdrs = true;
      else if (Tok == "AT")
        PhdrCmd.LMAExpr = readParenExpr();
      else if (Tok == "FLAGS") {
        expect("(");
        // Passing 0 for the value of dot is a bit of a hack. It means that
        // we accept expressions like ".|1".
        PhdrCmd.Flags = readExpr()(0);
        expect(")");
      } else
        setError("unexpected header attribute: " + Tok);
    } while (!Error);
  }
}

void ScriptParser::readSearchDir() {
  expect("(");
  StringRef Tok = next();
  if (!Config->Nostdlib)
    Config->SearchPaths.push_back(unquote(Tok));
  expect(")");
}

void ScriptParser::readSections() {
  Opt.HasSections = true;
  // -no-rosegment is used to avoid placing read only non-executable sections in
  // their own segment. We do the same if SECTIONS command is present in linker
  // script. See comment for computeFlags().
  Config->SingleRoRx = true;

  expect("{");
  while (!Error && !consume("}")) {
    StringRef Tok = next();
    BaseCommand *Cmd = readProvideOrAssignment(Tok);
    if (!Cmd) {
      if (Tok == "ASSERT")
        Cmd = new AssertCommand(readAssert());
      else
        Cmd = readOutputSectionDescription(Tok);
    }
    Opt.Commands.emplace_back(Cmd);
  }
}

static int precedence(StringRef Op) {
  return StringSwitch<int>(Op)
      .Cases("*", "/", 5)
      .Cases("+", "-", 4)
      .Cases("<<", ">>", 3)
      .Cases("<", "<=", ">", ">=", "==", "!=", 2)
      .Cases("&", "|", 1)
      .Default(-1);
}

StringMatcher ScriptParser::readFilePatterns() {
  std::vector<StringRef> V;
  while (!Error && !consume(")"))
    V.push_back(next());
  return StringMatcher(V);
}

SortSectionPolicy ScriptParser::readSortKind() {
  if (consume("SORT") || consume("SORT_BY_NAME"))
    return SortSectionPolicy::Name;
  if (consume("SORT_BY_ALIGNMENT"))
    return SortSectionPolicy::Alignment;
  if (consume("SORT_BY_INIT_PRIORITY"))
    return SortSectionPolicy::Priority;
  if (consume("SORT_NONE"))
    return SortSectionPolicy::None;
  return SortSectionPolicy::Default;
}

// Method reads a list of sequence of excluded files and section globs given in
// a following form: ((EXCLUDE_FILE(file_pattern+))? section_pattern+)+
// Example: *(.foo.1 EXCLUDE_FILE (*a.o) .foo.2 EXCLUDE_FILE (*b.o) .foo.3)
// The semantics of that is next:
// * Include .foo.1 from every file.
// * Include .foo.2 from every file but a.o
// * Include .foo.3 from every file but b.o
std::vector<SectionPattern> ScriptParser::readInputSectionsList() {
  std::vector<SectionPattern> Ret;
  while (!Error && peek() != ")") {
    StringMatcher ExcludeFilePat;
    if (consume("EXCLUDE_FILE")) {
      expect("(");
      ExcludeFilePat = readFilePatterns();
    }

    std::vector<StringRef> V;
    while (!Error && peek() != ")" && peek() != "EXCLUDE_FILE")
      V.push_back(next());

    if (!V.empty())
      Ret.push_back({std::move(ExcludeFilePat), StringMatcher(V)});
    else
      setError("section pattern is expected");
  }
  return Ret;
}

// Reads contents of "SECTIONS" directive. That directive contains a
// list of glob patterns for input sections. The grammar is as follows.
//
// <patterns> ::= <section-list>
//              | <sort> "(" <section-list> ")"
//              | <sort> "(" <sort> "(" <section-list> ")" ")"
//
// <sort>     ::= "SORT" | "SORT_BY_NAME" | "SORT_BY_ALIGNMENT"
//              | "SORT_BY_INIT_PRIORITY" | "SORT_NONE"
//
// <section-list> is parsed by readInputSectionsList().
InputSectionDescription *
ScriptParser::readInputSectionRules(StringRef FilePattern) {
  auto *Cmd = new InputSectionDescription(FilePattern);
  expect("(");
  while (!Error && !consume(")")) {
    SortSectionPolicy Outer = readSortKind();
    SortSectionPolicy Inner = SortSectionPolicy::Default;
    std::vector<SectionPattern> V;
    if (Outer != SortSectionPolicy::Default) {
      expect("(");
      Inner = readSortKind();
      if (Inner != SortSectionPolicy::Default) {
        expect("(");
        V = readInputSectionsList();
        expect(")");
      } else {
        V = readInputSectionsList();
      }
      expect(")");
    } else {
      V = readInputSectionsList();
    }

    for (SectionPattern &Pat : V) {
      Pat.SortInner = Inner;
      Pat.SortOuter = Outer;
    }

    std::move(V.begin(), V.end(), std::back_inserter(Cmd->SectionPatterns));
  }
  return Cmd;
}

InputSectionDescription *
ScriptParser::readInputSectionDescription(StringRef Tok) {
  // Input section wildcard can be surrounded by KEEP.
  // https://sourceware.org/binutils/docs/ld/Input-Section-Keep.html#Input-Section-Keep
  if (Tok == "KEEP") {
    expect("(");
    StringRef FilePattern = next();
    InputSectionDescription *Cmd = readInputSectionRules(FilePattern);
    expect(")");
    Opt.KeptSections.push_back(Cmd);
    return Cmd;
  }
  return readInputSectionRules(Tok);
}

void ScriptParser::readSort() {
  expect("(");
  expect("CONSTRUCTORS");
  expect(")");
}

Expr ScriptParser::readAssert() {
  expect("(");
  Expr E = readExpr();
  expect(",");
  StringRef Msg = unquote(next());
  expect(")");
  return [=](uint64_t Dot) {
    uint64_t V = E(Dot);
    if (!V)
      error(Msg);
    return V;
  };
}

// Reads a FILL(expr) command. We handle the FILL command as an
// alias for =fillexp section attribute, which is different from
// what GNU linkers do.
// https://sourceware.org/binutils/docs/ld/Output-Section-Data.html
uint32_t ScriptParser::readFill() {
  expect("(");
  uint32_t V = readOutputSectionFiller(next());
  expect(")");
  expect(";");
  return V;
}

OutputSectionCommand *
ScriptParser::readOutputSectionDescription(StringRef OutSec) {
  OutputSectionCommand *Cmd = new OutputSectionCommand(OutSec);
  Cmd->Location = getCurrentLocation();

  // Read an address expression.
  // https://sourceware.org/binutils/docs/ld/Output-Section-Address.html#Output-Section-Address
  if (peek() != ":")
    Cmd->AddrExpr = readExpr();

  expect(":");

  if (consume("AT"))
    Cmd->LMAExpr = readParenExpr();
  if (consume("ALIGN"))
    Cmd->AlignExpr = readParenExpr();
  if (consume("SUBALIGN"))
    Cmd->SubalignExpr = readParenExpr();

  // Parse constraints.
  if (consume("ONLY_IF_RO"))
    Cmd->Constraint = ConstraintKind::ReadOnly;
  if (consume("ONLY_IF_RW"))
    Cmd->Constraint = ConstraintKind::ReadWrite;
  expect("{");

  while (!Error && !consume("}")) {
    StringRef Tok = next();
    if (Tok == ";") {
      // Empty commands are allowed. Do nothing here.
    } else if (SymbolAssignment *Assignment = readProvideOrAssignment(Tok)) {
      Cmd->Commands.emplace_back(Assignment);
    } else if (BytesDataCommand *Data = readBytesDataCommand(Tok)) {
      Cmd->Commands.emplace_back(Data);
    } else if (Tok == "ASSERT") {
      Cmd->Commands.emplace_back(new AssertCommand(readAssert()));
      expect(";");
    } else if (Tok == "CONSTRUCTORS") {
      // CONSTRUCTORS is a keyword to make the linker recognize C++ ctors/dtors
      // by name. This is for very old file formats such as ECOFF/XCOFF.
      // For ELF, we should ignore.
    } else if (Tok == "FILL") {
      Cmd->Filler = readFill();
    } else if (Tok == "SORT") {
      readSort();
    } else if (peek() == "(") {
      Cmd->Commands.emplace_back(readInputSectionDescription(Tok));
    } else {
      setError("unknown command " + Tok);
    }
  }

  if (consume(">"))
    Cmd->MemoryRegionName = next();

  Cmd->Phdrs = readOutputSectionPhdrs();

  if (consume("="))
    Cmd->Filler = readOutputSectionFiller(next());
  else if (peek().startswith("="))
    Cmd->Filler = readOutputSectionFiller(next().drop_front());

  // Consume optional comma following output section command.
  consume(",");

  return Cmd;
}

// Read "=<number>" where <number> is an octal/decimal/hexadecimal number.
// https://sourceware.org/binutils/docs/ld/Output-Section-Fill.html
//
// ld.gold is not fully compatible with ld.bfd. ld.bfd handles
// hexstrings as blobs of arbitrary sizes, while ld.gold handles them
// as 32-bit big-endian values. We will do the same as ld.gold does
// because it's simpler than what ld.bfd does.
uint32_t ScriptParser::readOutputSectionFiller(StringRef Tok) {
  uint32_t V;
  if (!Tok.getAsInteger(0, V))
    return V;
  setError("invalid filler expression: " + Tok);
  return 0;
}

SymbolAssignment *ScriptParser::readProvideHidden(bool Provide, bool Hidden) {
  expect("(");
  SymbolAssignment *Cmd = readAssignment(next());
  Cmd->Provide = Provide;
  Cmd->Hidden = Hidden;
  expect(")");
  expect(";");
  return Cmd;
}

SymbolAssignment *ScriptParser::readProvideOrAssignment(StringRef Tok) {
  SymbolAssignment *Cmd = nullptr;
  if (peek() == "=" || peek() == "+=") {
    Cmd = readAssignment(Tok);
    expect(";");
  } else if (Tok == "PROVIDE") {
    Cmd = readProvideHidden(true, false);
  } else if (Tok == "HIDDEN") {
    Cmd = readProvideHidden(false, true);
  } else if (Tok == "PROVIDE_HIDDEN") {
    Cmd = readProvideHidden(true, true);
  }
  return Cmd;
}

static uint64_t getSymbolValue(const Twine &Loc, StringRef S, uint64_t Dot) {
  if (S == ".")
    return Dot;
  return ScriptBase->getSymbolValue(Loc, S);
}

static bool isAbsolute(StringRef S) {
  if (S == ".")
    return false;
  return ScriptBase->isAbsolute(S);
}

SymbolAssignment *ScriptParser::readAssignment(StringRef Name) {
  StringRef Op = next();
  Expr E;
  assert(Op == "=" || Op == "+=");
  if (consume("ABSOLUTE")) {
    E = readExpr();
    E.IsAbsolute = [] { return true; };
  } else {
    E = readExpr();
  }
  if (Op == "+=") {
    std::string Loc = getCurrentLocation();
    E = [=](uint64_t Dot) {
      return getSymbolValue(Loc, Name, Dot) + E(Dot);
    };
  }
  return new SymbolAssignment(Name, E);
}

// This is an operator-precedence parser to parse a linker
// script expression.
Expr ScriptParser::readExpr() {
  // Our lexer is context-aware. Set the in-expression bit so that
  // they apply different tokenization rules.
  bool Orig = InExpr;
  InExpr = true;
  Expr E = readExpr1(readPrimary(), 0);
  InExpr = Orig;
  return E;
}

static Expr combine(StringRef Op, Expr L, Expr R) {
  auto IsAbs = [=] { return L.IsAbsolute() && R.IsAbsolute(); };
  auto GetOutSec = [=] {
    const OutputSectionBase *S = L.Section();
    return S ? S : R.Section();
  };

  if (Op == "*")
    return [=](uint64_t Dot) { return L(Dot) * R(Dot); };
  if (Op == "/") {
    return [=](uint64_t Dot) -> uint64_t {
      uint64_t RHS = R(Dot);
      if (RHS == 0) {
        error("division by zero");
        return 0;
      }
      return L(Dot) / RHS;
    };
  }
  if (Op == "+")
    return {[=](uint64_t Dot) { return L(Dot) + R(Dot); }, IsAbs, GetOutSec};
  if (Op == "-")
    return {[=](uint64_t Dot) { return L(Dot) - R(Dot); }, IsAbs, GetOutSec};
  if (Op == "<<")
    return [=](uint64_t Dot) { return L(Dot) << R(Dot); };
  if (Op == ">>")
    return [=](uint64_t Dot) { return L(Dot) >> R(Dot); };
  if (Op == "<")
    return [=](uint64_t Dot) { return L(Dot) < R(Dot); };
  if (Op == ">")
    return [=](uint64_t Dot) { return L(Dot) > R(Dot); };
  if (Op == ">=")
    return [=](uint64_t Dot) { return L(Dot) >= R(Dot); };
  if (Op == "<=")
    return [=](uint64_t Dot) { return L(Dot) <= R(Dot); };
  if (Op == "==")
    return [=](uint64_t Dot) { return L(Dot) == R(Dot); };
  if (Op == "!=")
    return [=](uint64_t Dot) { return L(Dot) != R(Dot); };
  if (Op == "&")
    return [=](uint64_t Dot) { return L(Dot) & R(Dot); };
  if (Op == "|")
    return [=](uint64_t Dot) { return L(Dot) | R(Dot); };
  llvm_unreachable("invalid operator");
}

// This is a part of the operator-precedence parser. This function
// assumes that the remaining token stream starts with an operator.
Expr ScriptParser::readExpr1(Expr Lhs, int MinPrec) {
  while (!atEOF() && !Error) {
    // Read an operator and an expression.
    if (consume("?"))
      return readTernary(Lhs);
    StringRef Op1 = peek();
    if (precedence(Op1) < MinPrec)
      break;
    skip();
    Expr Rhs = readPrimary();

    // Evaluate the remaining part of the expression first if the
    // next operator has greater precedence than the previous one.
    // For example, if we have read "+" and "3", and if the next
    // operator is "*", then we'll evaluate 3 * ... part first.
    while (!atEOF()) {
      StringRef Op2 = peek();
      if (precedence(Op2) <= precedence(Op1))
        break;
      Rhs = readExpr1(Rhs, precedence(Op2));
    }

    Lhs = combine(Op1, Lhs, Rhs);
  }
  return Lhs;
}

uint64_t static getConstant(StringRef S) {
  if (S == "COMMONPAGESIZE")
    return Target->PageSize;
  if (S == "MAXPAGESIZE")
    return Config->MaxPageSize;
  error("unknown constant: " + S);
  return 0;
}

// Parses Tok as an integer. Returns true if successful.
// It recognizes hexadecimal (prefixed with "0x" or suffixed with "H")
// and decimal numbers. Decimal numbers may have "K" (kilo) or
// "M" (mega) prefixes.
static bool readInteger(StringRef Tok, uint64_t &Result) {
  // Negative number
  if (Tok.startswith("-")) {
    if (!readInteger(Tok.substr(1), Result))
      return false;
    Result = -Result;
    return true;
  }

  // Hexadecimal
  if (Tok.startswith_lower("0x"))
    return !Tok.substr(2).getAsInteger(16, Result);
  if (Tok.endswith_lower("H"))
    return !Tok.drop_back().getAsInteger(16, Result);

  // Decimal
  int Suffix = 1;
  if (Tok.endswith_lower("K")) {
    Suffix = 1024;
    Tok = Tok.drop_back();
  } else if (Tok.endswith_lower("M")) {
    Suffix = 1024 * 1024;
    Tok = Tok.drop_back();
  }
  if (Tok.getAsInteger(10, Result))
    return false;
  Result *= Suffix;
  return true;
}

BytesDataCommand *ScriptParser::readBytesDataCommand(StringRef Tok) {
  int Size = StringSwitch<unsigned>(Tok)
                 .Case("BYTE", 1)
                 .Case("SHORT", 2)
                 .Case("LONG", 4)
                 .Case("QUAD", 8)
                 .Default(-1);
  if (Size == -1)
    return nullptr;

  return new BytesDataCommand(readParenExpr(), Size);
}

StringRef ScriptParser::readParenLiteral() {
  expect("(");
  StringRef Tok = next();
  expect(")");
  return Tok;
}

Expr ScriptParser::readPrimary() {
  if (peek() == "(")
    return readParenExpr();

  StringRef Tok = next();
  std::string Location = getCurrentLocation();

  if (Tok == "~") {
    Expr E = readPrimary();
    return [=](uint64_t Dot) { return ~E(Dot); };
  }
  if (Tok == "-") {
    Expr E = readPrimary();
    return [=](uint64_t Dot) { return -E(Dot); };
  }

  // Built-in functions are parsed here.
  // https://sourceware.org/binutils/docs/ld/Builtin-Functions.html.
  if (Tok == "ADDR") {
    StringRef Name = readParenLiteral();
    return {[=](uint64_t Dot) {
              return ScriptBase->getOutputSection(Location, Name)->Addr;
            },
            [=] { return false; },
            [=] { return ScriptBase->getOutputSection(Location, Name); }};
  }
  if (Tok == "LOADADDR") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) {
      return ScriptBase->getOutputSection(Location, Name)->getLMA();
    };
  }
  if (Tok == "ASSERT")
    return readAssert();
  if (Tok == "ALIGN") {
    expect("(");
    Expr E = readExpr();
    if (consume(",")) {
      Expr E2 = readExpr();
      expect(")");
      return [=](uint64_t Dot) { return alignTo(E(Dot), E2(Dot)); };
    }
    expect(")");
    return [=](uint64_t Dot) { return alignTo(Dot, E(Dot)); };
  }
  if (Tok == "CONSTANT") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) { return getConstant(Name); };
  }
  if (Tok == "DEFINED") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) { return ScriptBase->isDefined(Name) ? 1 : 0; };
  }
  if (Tok == "SEGMENT_START") {
    expect("(");
    skip();
    expect(",");
    Expr E = readExpr();
    expect(")");
    return [=](uint64_t Dot) { return E(Dot); };
  }
  if (Tok == "DATA_SEGMENT_ALIGN") {
    expect("(");
    Expr E = readExpr();
    expect(",");
    readExpr();
    expect(")");
    return [=](uint64_t Dot) { return alignTo(Dot, E(Dot)); };
  }
  if (Tok == "DATA_SEGMENT_END") {
    expect("(");
    expect(".");
    expect(")");
    return [](uint64_t Dot) { return Dot; };
  }
  // GNU linkers implements more complicated logic to handle
  // DATA_SEGMENT_RELRO_END. We instead ignore the arguments and just align to
  // the next page boundary for simplicity.
  if (Tok == "DATA_SEGMENT_RELRO_END") {
    expect("(");
    readExpr();
    expect(",");
    readExpr();
    expect(")");
    return [](uint64_t Dot) { return alignTo(Dot, Target->PageSize); };
  }
  if (Tok == "SIZEOF") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) { return ScriptBase->getOutputSectionSize(Name); };
  }
  if (Tok == "ALIGNOF") {
    StringRef Name = readParenLiteral();
    return [=](uint64_t Dot) {
      return ScriptBase->getOutputSection(Location, Name)->Addralign;
    };
  }
  if (Tok == "SIZEOF_HEADERS")
    return [=](uint64_t Dot) { return ScriptBase->getHeaderSize(); };

  // Tok is a literal number.
  uint64_t V;
  if (readInteger(Tok, V))
    return [=](uint64_t Dot) { return V; };

  // Tok is a symbol name.
  if (Tok != "." && !isValidCIdentifier(Tok))
    setError("malformed number: " + Tok);
  return {[=](uint64_t Dot) { return getSymbolValue(Location, Tok, Dot); },
          [=] { return isAbsolute(Tok); },
          [=] { return ScriptBase->getSymbolSection(Tok); }};
}

Expr ScriptParser::readTernary(Expr Cond) {
  Expr L = readExpr();
  expect(":");
  Expr R = readExpr();
  return [=](uint64_t Dot) { return Cond(Dot) ? L(Dot) : R(Dot); };
}

Expr ScriptParser::readParenExpr() {
  expect("(");
  Expr E = readExpr();
  expect(")");
  return E;
}

std::vector<StringRef> ScriptParser::readOutputSectionPhdrs() {
  std::vector<StringRef> Phdrs;
  while (!Error && peek().startswith(":")) {
    StringRef Tok = next();
    Phdrs.push_back((Tok.size() == 1) ? next() : Tok.substr(1));
  }
  return Phdrs;
}

// Read a program header type name. The next token must be a
// name of a program header type or a constant (e.g. "0x3").
unsigned ScriptParser::readPhdrType() {
  StringRef Tok = next();
  uint64_t Val;
  if (readInteger(Tok, Val))
    return Val;

  unsigned Ret = StringSwitch<unsigned>(Tok)
                     .Case("PT_NULL", PT_NULL)
                     .Case("PT_LOAD", PT_LOAD)
                     .Case("PT_DYNAMIC", PT_DYNAMIC)
                     .Case("PT_INTERP", PT_INTERP)
                     .Case("PT_NOTE", PT_NOTE)
                     .Case("PT_SHLIB", PT_SHLIB)
                     .Case("PT_PHDR", PT_PHDR)
                     .Case("PT_TLS", PT_TLS)
                     .Case("PT_GNU_EH_FRAME", PT_GNU_EH_FRAME)
                     .Case("PT_GNU_STACK", PT_GNU_STACK)
                     .Case("PT_GNU_RELRO", PT_GNU_RELRO)
                     .Case("PT_OPENBSD_RANDOMIZE", PT_OPENBSD_RANDOMIZE)
                     .Case("PT_OPENBSD_WXNEEDED", PT_OPENBSD_WXNEEDED)
                     .Case("PT_OPENBSD_BOOTDATA", PT_OPENBSD_BOOTDATA)
                     .Default(-1);

  if (Ret == (unsigned)-1) {
    setError("invalid program header type: " + Tok);
    return PT_NULL;
  }
  return Ret;
}

// Reads a list of symbols, e.g. "{ global: foo; bar; local: *; };".
void ScriptParser::readAnonymousDeclaration() {
  // Read global symbols first. "global:" is default, so if there's
  // no label, we assume global symbols.
  if (peek() != "local") {
    if (consume("global"))
      expect(":");
    for (SymbolVersion V : readSymbols())
      Config->VersionScriptGlobals.push_back(V);
  }
  readLocals();
  expect("}");
  expect(";");
}

void ScriptParser::readLocals() {
  if (!consume("local"))
    return;
  expect(":");
  std::vector<SymbolVersion> Locals = readSymbols();
  for (SymbolVersion V : Locals) {
    if (V.Name == "*") {
      Config->DefaultSymbolVersion = VER_NDX_LOCAL;
      continue;
    }
    Config->VersionScriptLocals.push_back(V);
  }
}

// Reads a list of symbols, e.g. "VerStr { global: foo; bar; local: *; };".
void ScriptParser::readVersionDeclaration(StringRef VerStr) {
  // Identifiers start at 2 because 0 and 1 are reserved
  // for VER_NDX_LOCAL and VER_NDX_GLOBAL constants.
  uint16_t VersionId = Config->VersionDefinitions.size() + 2;
  Config->VersionDefinitions.push_back({VerStr, VersionId});

  // Read global symbols.
  if (peek() != "local") {
    if (consume("global"))
      expect(":");
    Config->VersionDefinitions.back().Globals = readSymbols();
  }
  readLocals();
  expect("}");

  // Each version may have a parent version. For example, "Ver2"
  // defined as "Ver2 { global: foo; local: *; } Ver1;" has "Ver1"
  // as a parent. This version hierarchy is, probably against your
  // instinct, purely for hint; the runtime doesn't care about it
  // at all. In LLD, we simply ignore it.
  if (peek() != ";")
    skip();
  expect(";");
}

// Reads a list of symbols for a versions cript.
std::vector<SymbolVersion> ScriptParser::readSymbols() {
  std::vector<SymbolVersion> Ret;
  for (;;) {
    if (consume("extern")) {
      for (SymbolVersion V : readVersionExtern())
        Ret.push_back(V);
      continue;
    }

    if (peek() == "}" || (peek() == "local" && peek(1) == ":") || Error)
      break;
    StringRef Tok = next();
    Ret.push_back({unquote(Tok), false, hasWildcard(Tok)});
    expect(";");
  }
  return Ret;
}

// Reads an "extern C++" directive, e.g.,
// "extern "C++" { ns::*; "f(int, double)"; };"
std::vector<SymbolVersion> ScriptParser::readVersionExtern() {
  StringRef Tok = next();
  bool IsCXX = Tok == "\"C++\"";
  if (!IsCXX && Tok != "\"C\"")
    setError("Unknown language");
  expect("{");

  std::vector<SymbolVersion> Ret;
  while (!Error && peek() != "}") {
    StringRef Tok = next();
    bool HasWildcard = !Tok.startswith("\"") && hasWildcard(Tok);
    Ret.push_back({unquote(Tok), IsCXX, HasWildcard});
    expect(";");
  }

  expect("}");
  expect(";");
  return Ret;
}

uint64_t ScriptParser::readMemoryAssignment(
    StringRef S1, StringRef S2, StringRef S3) {
  if (!(consume(S1) || consume(S2) || consume(S3))) {
    setError("expected one of: " + S1 + ", " + S2 + ", or " + S3);
    return 0;
  }
  expect("=");

  // TODO: Fully support constant expressions.
  uint64_t Val;
  if (!readInteger(next(), Val))
    setError("nonconstant expression for "+ S1);
  return Val;
}

// Parse the MEMORY command as specified in:
// https://sourceware.org/binutils/docs/ld/MEMORY.html
//
// MEMORY { name [(attr)] : ORIGIN = origin, LENGTH = len ... }
void ScriptParser::readMemory() {
  expect("{");
  while (!Error && !consume("}")) {
    StringRef Name = next();

    uint32_t Flags = 0;
    uint32_t NegFlags = 0;
    if (consume("(")) {
      std::tie(Flags, NegFlags) = readMemoryAttributes();
      expect(")");
    }
    expect(":");

    uint64_t Origin = readMemoryAssignment("ORIGIN", "org", "o");
    expect(",");
    uint64_t Length = readMemoryAssignment("LENGTH", "len", "l");

    // Add the memory region to the region map (if it doesn't already exist).
    auto It = Opt.MemoryRegions.find(Name);
    if (It != Opt.MemoryRegions.end())
      setError("region '" + Name + "' already defined");
    else
      Opt.MemoryRegions[Name] = {Name, Origin, Length, Origin, Flags, NegFlags};
  }
}

// This function parses the attributes used to match against section
// flags when placing output sections in a memory region. These flags
// are only used when an explicit memory region name is not used.
std::pair<uint32_t, uint32_t> ScriptParser::readMemoryAttributes() {
  uint32_t Flags = 0;
  uint32_t NegFlags = 0;
  bool Invert = false;

  for (char C : next().lower()) {
    uint32_t Flag = 0;
    if (C == '!')
      Invert = !Invert;
    else if (C == 'w')
      Flag = SHF_WRITE;
    else if (C == 'x')
      Flag = SHF_EXECINSTR;
    else if (C == 'a')
      Flag = SHF_ALLOC;
    else if (C != 'r')
      setError("invalid memory region attribute");

    if (Invert)
      NegFlags |= Flag;
    else
      Flags |= Flag;
  }
  return {Flags, NegFlags};
}

void elf::readLinkerScript(MemoryBufferRef MB) {
  ScriptParser(MB).readLinkerScript();
}

void elf::readVersionScript(MemoryBufferRef MB) {
  ScriptParser(MB).readVersionScript();
}

void elf::readDynamicList(MemoryBufferRef MB) {
  ScriptParser(MB).readDynamicList();
}

template class elf::LinkerScript<ELF32LE>;
template class elf::LinkerScript<ELF32BE>;
template class elf::LinkerScript<ELF64LE>;
template class elf::LinkerScript<ELF64BE>;
