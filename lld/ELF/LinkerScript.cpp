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
// It parses a linker script and write the result to Config or ScriptConfig
// objects.
//
// If SECTIONS command is used, a ScriptConfig contains an AST
// of the command which will later be consumed by createSections() and
// assignAddresses().
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "Driver.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "ScriptParser.h"
#include "Strings.h"
#include "Symbols.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Writer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace lld;
using namespace lld::elf;

LinkerScriptBase *elf::ScriptBase;
ScriptConfiguration *elf::ScriptConfig;

template <class ELFT> static void addRegular(SymbolAssignment *Cmd) {
  Symbol *Sym = Symtab<ELFT>::X->addRegular(Cmd->Name, STB_GLOBAL, STV_DEFAULT);
  Sym->Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  Cmd->Sym = Sym->body();

  // If we have no SECTIONS then we don't have '.' and don't call
  // assignAddresses(). We calculate symbol value immediately in this case.
  if (!ScriptConfig->HasSections)
    cast<DefinedRegular<ELFT>>(Cmd->Sym)->Value = Cmd->Expression(0);
}

template <class ELFT> static void addSynthetic(SymbolAssignment *Cmd) {
  Symbol *Sym = Symtab<ELFT>::X->addSynthetic(
      Cmd->Name, nullptr, 0, Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT);
  Cmd->Sym = Sym->body();
}

template <class ELFT> static void addSymbol(SymbolAssignment *Cmd) {
  if (Cmd->IsAbsolute)
    addRegular<ELFT>(Cmd);
  else
    addSynthetic<ELFT>(Cmd);
}
// If a symbol was in PROVIDE(), we need to define it only when
// it is an undefined symbol.
template <class ELFT> static bool shouldDefine(SymbolAssignment *Cmd) {
  if (Cmd->Name == ".")
    return false;
  if (!Cmd->Provide)
    return true;
  SymbolBody *B = Symtab<ELFT>::X->find(Cmd->Name);
  return B && B->isUndefined();
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

template <class ELFT> static bool isDiscarded(InputSectionBase<ELFT> *S) {
  return !S || !S->Live;
}

template <class ELFT> LinkerScript<ELFT>::LinkerScript() {}
template <class ELFT> LinkerScript<ELFT>::~LinkerScript() {}

template <class ELFT>
bool LinkerScript<ELFT>::shouldKeep(InputSectionBase<ELFT> *S) {
  for (Regex *Re : Opt.KeptSections)
    if (Re->match(S->Name))
      return true;
  return false;
}

// We need to use const_cast because match() is not a const function.
// This function encapsulates that ugliness.
static bool match(const Regex &Re, StringRef S) {
  return const_cast<Regex &>(Re).match(S);
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

static bool checkConstraint(uint64_t Flags, ConstraintKind Kind) {
  bool RO = (Kind == ConstraintKind::ReadOnly);
  bool RW = (Kind == ConstraintKind::ReadWrite);
  bool Writable = Flags & SHF_WRITE;
  return !(RO && Writable) && !(RW && !Writable);
}

template <class ELFT>
static bool matchConstraints(ArrayRef<InputSectionBase<ELFT> *> Sections,
                             ConstraintKind Kind) {
  if (Kind == ConstraintKind::NoConstraint)
    return true;
  return llvm::all_of(Sections, [=](InputSectionData *Sec2) {
    auto *Sec = static_cast<InputSectionBase<ELFT> *>(Sec2);
    return checkConstraint(Sec->getSectionHdr()->sh_flags, Kind);
  });
}

// Compute and remember which sections the InputSectionDescription matches.
template <class ELFT>
void LinkerScript<ELFT>::computeInputSections(InputSectionDescription *I) {
  // Collects all sections that satisfy constraints of I
  // and attach them to I.
  for (SectionPattern &Pat : I->SectionPatterns) {
    for (ObjectFile<ELFT> *F : Symtab<ELFT>::X->getObjectFiles()) {
      StringRef Filename = sys::path::filename(F->getName());
      if (!match(I->FileRe, Filename) || match(Pat.ExcludedFileRe, Filename))
        continue;

      for (InputSectionBase<ELFT> *S : F->getSections())
        if (!isDiscarded(S) && !S->OutSec && match(Pat.SectionRe, S->Name))
          I->Sections.push_back(S);
      if (match(Pat.SectionRe, "COMMON"))
        I->Sections.push_back(CommonInputSection<ELFT>::X);
    }
  }

  // Sort for SORT() commands.
  if (I->SortInner != SortSectionPolicy::Default)
    std::stable_sort(I->Sections.begin(), I->Sections.end(),
                     getComparator(I->SortInner));
  if (I->SortOuter != SortSectionPolicy::Default)
    std::stable_sort(I->Sections.begin(), I->Sections.end(),
                     getComparator(I->SortOuter));

  // We do not add duplicate input sections, so mark them with a dummy output
  // section for now.
  for (InputSectionData *S : I->Sections) {
    auto *S2 = static_cast<InputSectionBase<ELFT> *>(S);
    S2->OutSec = (OutputSectionBase<ELFT> *)-1;
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
static SectionKey<ELFT::Is64Bits> createKey(InputSectionBase<ELFT> *C,
                                            StringRef OutsecName) {
  // When using linker script the merge rules are different.
  // Unfortunately, linker scripts are name based. This means that expressions
  // like *(.foo*) can refer to multiple input sections that would normally be
  // placed in different output sections. We cannot put them in different
  // output sections or we would produce wrong results for
  // start = .; *(.foo.*) end = .; *(.bar)
  // and a mapping of .foo1 and .bar1 to one section and .foo2 and .bar2 to
  // another. The problem is that there is no way to layout those output
  // sections such that the .foo sections are the only thing between the
  // start and end symbols.

  // An extra annoyance is that we cannot simply disable merging of the contents
  // of SHF_MERGE sections, but our implementation requires one output section
  // per "kind" (string or not, which size/aligment).
  // Fortunately, creating symbols in the middle of a merge section is not
  // supported by bfd or gold, so we can just create multiple section in that
  // case.
  const typename ELFT::Shdr *H = C->getSectionHdr();
  typedef typename ELFT::uint uintX_t;
  uintX_t Flags = H->sh_flags & (SHF_MERGE | SHF_STRINGS);

  uintX_t Alignment = 0;
  if (isa<MergeInputSection<ELFT>>(C))
    Alignment = std::max(H->sh_addralign, H->sh_entsize);

  return SectionKey<ELFT::Is64Bits>{OutsecName, /*Type*/ 0, Flags, Alignment};
}

template <class ELFT>
void LinkerScript<ELFT>::addSection(OutputSectionFactory<ELFT> &Factory,
                                    InputSectionBase<ELFT> *Sec,
                                    StringRef Name) {
  OutputSectionBase<ELFT> *OutSec;
  bool IsNew;
  std::tie(OutSec, IsNew) = Factory.create(createKey(Sec, Name), Sec);
  if (IsNew)
    OutputSections->push_back(OutSec);
  OutSec->addSection(Sec);
}

template <class ELFT>
void LinkerScript<ELFT>::processCommands(OutputSectionFactory<ELFT> &Factory) {

  for (unsigned I = 0; I < Opt.Commands.size(); ++I) {
    auto Iter = Opt.Commands.begin() + I;
    const std::unique_ptr<BaseCommand> &Base1 = *Iter;
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base1.get())) {
      if (shouldDefine<ELFT>(Cmd))
        addRegular<ELFT>(Cmd);
      continue;
    }
    if (auto *Cmd = dyn_cast<AssertCommand>(Base1.get())) {
      // If we don't have SECTIONS then output sections have already been
      // created by Writer<EFLT>. The LinkerScript<ELFT>::assignAddresses
      // will not be called, so ASSERT should be evaluated now.
      if (!Opt.HasSections)
        Cmd->Expression(0);
      continue;
    }

    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base1.get())) {
      std::vector<InputSectionBase<ELFT> *> V = createInputSectionList(*Cmd);

      if (Cmd->Name == "/DISCARD/") {
        discard(V);
        continue;
      }

      if (!matchConstraints<ELFT>(V, Cmd->Constraint)) {
        for (InputSectionBase<ELFT> *S : V)
          S->OutSec = nullptr;
        Opt.Commands.erase(Iter);
        --I;
        continue;
      }

      for (const std::unique_ptr<BaseCommand> &Base : Cmd->Commands)
        if (auto *OutCmd = dyn_cast<SymbolAssignment>(Base.get()))
          if (shouldDefine<ELFT>(OutCmd))
            addSymbol<ELFT>(OutCmd);

      if (V.empty())
        continue;

      for (InputSectionBase<ELFT> *Sec : V) {
        addSection(Factory, Sec, Cmd->Name);
        if (uint32_t Subalign = Cmd->SubalignExpr ? Cmd->SubalignExpr(0) : 0)
          Sec->Alignment = Subalign;
      }
    }
  }
}

template <class ELFT>
void LinkerScript<ELFT>::createSections(OutputSectionFactory<ELFT> &Factory) {
  processCommands(Factory);
  // Add orphan sections.
  for (ObjectFile<ELFT> *F : Symtab<ELFT>::X->getObjectFiles())
    for (InputSectionBase<ELFT> *S : F->getSections())
      if (!isDiscarded(S) && !S->OutSec)
        addSection(Factory, S, getOutputSectionName(S));
}

// Sets value of a section-defined symbol. Two kinds of
// symbols are processed: synthetic symbols, whose value
// is an offset from beginning of section and regular
// symbols whose value is absolute.
template <class ELFT>
static void assignSectionSymbol(SymbolAssignment *Cmd,
                                OutputSectionBase<ELFT> *Sec,
                                typename ELFT::uint Off) {
  if (!Cmd->Sym)
    return;

  if (auto *Body = dyn_cast<DefinedSynthetic<ELFT>>(Cmd->Sym)) {
    Body->Section = Sec;
    Body->Value = Cmd->Expression(Sec->getVA() + Off) - Sec->getVA();
    return;
  }
  auto *Body = cast<DefinedRegular<ELFT>>(Cmd->Sym);
  Body->Value = Cmd->Expression(Sec->getVA() + Off);
}

template <class ELFT> void LinkerScript<ELFT>::output(InputSection<ELFT> *S) {
  if (!AlreadyOutputIS.insert(S).second)
    return;
  bool IsTbss =
      (CurOutSec->getFlags() & SHF_TLS) && CurOutSec->getType() == SHT_NOBITS;

  uintX_t Pos = IsTbss ? Dot + ThreadBssOffset : Dot;
  Pos = alignTo(Pos, S->Alignment);
  S->OutSecOff = Pos - CurOutSec->getVA();
  Pos += S->getSize();

  // Update output section size after adding each section. This is so that
  // SIZEOF works correctly in the case below:
  // .foo { *(.aaa) a = SIZEOF(.foo); *(.bbb) }
  CurOutSec->setSize(Pos - CurOutSec->getVA());

  if (!IsTbss)
    Dot = Pos;
}

template <class ELFT> void LinkerScript<ELFT>::flush() {
  if (auto *OutSec = dyn_cast_or_null<OutputSection<ELFT>>(CurOutSec)) {
    for (InputSection<ELFT> *I : OutSec->Sections)
      output(I);
    AlreadyOutputOS.insert(CurOutSec);
  }
}

template <class ELFT>
void LinkerScript<ELFT>::switchTo(OutputSectionBase<ELFT> *Sec) {
  if (CurOutSec == Sec)
    return;
  if (AlreadyOutputOS.count(Sec))
    return;

  flush();
  CurOutSec = Sec;

  Dot = alignTo(Dot, CurOutSec->getAlignment());
  CurOutSec->setVA(Dot);
}

template <class ELFT> void LinkerScript<ELFT>::process(BaseCommand &Base) {
  if (auto *AssignCmd = dyn_cast<SymbolAssignment>(&Base)) {
    if (AssignCmd->Name == ".") {
      // Update to location counter means update to section size.
      Dot = AssignCmd->Expression(Dot);
      CurOutSec->setSize(Dot - CurOutSec->getVA());
      return;
    }
    assignSectionSymbol<ELFT>(AssignCmd, CurOutSec, Dot - CurOutSec->getVA());
    return;
  }
  auto &ICmd = cast<InputSectionDescription>(Base);
  for (InputSectionData *ID : ICmd.Sections) {
    auto *IB = static_cast<InputSectionBase<ELFT> *>(ID);
    switchTo(IB->OutSec);
    if (auto *I = dyn_cast<InputSection<ELFT>>(IB))
      output(I);
    else if (AlreadyOutputOS.insert(CurOutSec).second)
      Dot += CurOutSec->getSize();
  }
}

template <class ELFT>
static std::vector<OutputSectionBase<ELFT> *>
findSections(OutputSectionCommand &Cmd,
             const std::vector<OutputSectionBase<ELFT> *> &Sections) {
  std::vector<OutputSectionBase<ELFT> *> Ret;
  for (OutputSectionBase<ELFT> *Sec : Sections)
    if (Sec->getName() == Cmd.Name)
      Ret.push_back(Sec);
  return Ret;
}

template <class ELFT>
void LinkerScript<ELFT>::assignOffsets(OutputSectionCommand *Cmd) {
  std::vector<OutputSectionBase<ELFT> *> Sections =
      findSections(*Cmd, *OutputSections);
  if (Sections.empty())
    return;
  switchTo(Sections[0]);

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
  for (OutputSectionBase<ELFT> *Base : Sections) {
    if (!AlreadyOutputOS.insert(Base).second)
      continue;
    switchTo(Base);
    Dot += CurOutSec->getSize();
  }
  for (auto I = E, E = Cmd->Commands.end(); I != E; ++I)
    process(**I);
}

template <class ELFT> void LinkerScript<ELFT>::assignAddresses() {
  // Orphan sections are sections present in the input files which
  // are not explicitly placed into the output file by the linker script.
  // We place orphan sections at end of file.
  // Other linkers places them using some heuristics as described in
  // https://sourceware.org/binutils/docs/ld/Orphan-Sections.html#Orphan-Sections.

  // The OutputSections are already in the correct order.
  // This loops creates or moves commands as needed so that they are in the
  // correct order.
  int CmdIndex = 0;
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    StringRef Name = Sec->getName();

    // Find the last spot where we can insert a command and still get the
    // correct order.
    auto CmdIter = Opt.Commands.begin() + CmdIndex;
    auto E = Opt.Commands.end();
    while (CmdIter != E && !isa<OutputSectionCommand>(**CmdIter)) {
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
    } else {
      // If linker script lists alloc/non-alloc sections is the wrong order,
      // this does a right rotate to bring the desired command in place.
      auto RPos = llvm::make_reverse_iterator(Pos + 1);
      std::rotate(RPos, RPos + 1, llvm::make_reverse_iterator(CmdIter));
    }
    ++CmdIndex;
  }

  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = getHeaderSize();

  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base.get())) {
      if (Cmd->Name == ".") {
        Dot = Cmd->Expression(Dot);
      } else if (Cmd->Sym) {
        cast<DefinedRegular<ELFT>>(Cmd->Sym)->Value = Cmd->Expression(Dot);
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
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    if (Sec->getFlags() & SHF_ALLOC)
      MinVA = std::min(MinVA, Sec->getVA());
    else
      Sec->setVA(0);
  }

  uintX_t HeaderSize =
      Out<ELFT>::ElfHeader->getSize() + Out<ELFT>::ProgramHeaders->getSize();
  if (HeaderSize > MinVA)
    fatal("Not enough space for ELF and program headers");

  // ELF and Program headers need to be right before the first section in
  // memory. Set their addresses accordingly.
  MinVA = alignDown(MinVA - HeaderSize, Target->PageSize);
  Out<ELFT>::ElfHeader->setVA(MinVA);
  Out<ELFT>::ProgramHeaders->setVA(Out<ELFT>::ElfHeader->getSize() + MinVA);
}

// Creates program headers as instructed by PHDRS linker script command.
template <class ELFT>
std::vector<PhdrEntry<ELFT>> LinkerScript<ELFT>::createPhdrs() {
  std::vector<PhdrEntry<ELFT>> Ret;

  // Process PHDRS and FILEHDR keywords because they are not
  // real output sections and cannot be added in the following loop.
  for (const PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    Ret.emplace_back(Cmd.Type, Cmd.Flags == UINT_MAX ? PF_R : Cmd.Flags);
    PhdrEntry<ELFT> &Phdr = Ret.back();

    if (Cmd.HasFilehdr)
      Phdr.add(Out<ELFT>::ElfHeader);
    if (Cmd.HasPhdrs)
      Phdr.add(Out<ELFT>::ProgramHeaders);

    if (Cmd.LMAExpr) {
      Phdr.H.p_paddr = Cmd.LMAExpr(0);
      Phdr.HasLMA = true;
    }
  }

  // Add output sections to program headers.
  PhdrEntry<ELFT> *Load = nullptr;
  uintX_t Flags = PF_R;
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    if (!(Sec->getFlags() & SHF_ALLOC))
      break;

    std::vector<size_t> PhdrIds = getPhdrIndices(Sec->getName());
    if (!PhdrIds.empty()) {
      // Assign headers specified by linker script
      for (size_t Id : PhdrIds) {
        Ret[Id].add(Sec);
        if (Opt.PhdrsCommands[Id].Flags == UINT_MAX)
          Ret[Id].H.p_flags |= Sec->getPhdrFlags();
      }
    } else {
      // If we have no load segment or flags've changed then we want new load
      // segment.
      uintX_t NewFlags = Sec->getPhdrFlags();
      if (Load == nullptr || Flags != NewFlags) {
        Load = &*Ret.emplace(Ret.end(), PT_LOAD, NewFlags);
        Flags = NewFlags;
      }
      Load->add(Sec);
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

template <class ELFT>
ArrayRef<uint8_t> LinkerScript<ELFT>::getFiller(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->Name == Name)
        return Cmd->Filler;
  return {};
}

template <class ELFT> Expr LinkerScript<ELFT>::getLma(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->LmaExpr && Cmd->Name == Name)
        return Cmd->LmaExpr;
  return {};
}

// Returns the index of the given section name in linker script
// SECTIONS commands. Sections are laid out as the same order as they
// were in the script. If a given name did not appear in the script,
// it returns INT_MAX, so that it will be laid out at end of file.
template <class ELFT> int LinkerScript<ELFT>::getSectionIndex(StringRef Name) {
  int I = 0;
  for (std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->Name == Name)
        return I;
    ++I;
  }
  return INT_MAX;
}

// A compartor to sort output sections. Returns -1 or 1 if
// A or B are mentioned in linker script. Otherwise, returns 0.
template <class ELFT>
int LinkerScript<ELFT>::compareSections(StringRef A, StringRef B) {
  int I = getSectionIndex(A);
  int J = getSectionIndex(B);
  if (I == INT_MAX && J == INT_MAX)
    return 0;
  return I < J ? -1 : 1;
}

template <class ELFT> bool LinkerScript<ELFT>::hasPhdrsCommands() {
  return !Opt.PhdrsCommands.empty();
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionAddress(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getVA();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionSize(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getSize();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT>
uint64_t LinkerScript<ELFT>::getOutputSectionAlign(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getAlignment();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT> uint64_t LinkerScript<ELFT>::getHeaderSize() {
  return Out<ELFT>::ElfHeader->getSize() + Out<ELFT>::ProgramHeaders->getSize();
}

template <class ELFT> uint64_t LinkerScript<ELFT>::getSymbolValue(StringRef S) {
  if (SymbolBody *B = Symtab<ELFT>::X->find(S))
    return B->getVA<ELFT>();
  error("symbol not found: " + S);
  return 0;
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
      Ret.push_back(getPhdrIndex(PhdrName));
    return Ret;
  }
  return {};
}

template <class ELFT>
size_t LinkerScript<ELFT>::getPhdrIndex(StringRef PhdrName) {
  size_t I = 0;
  for (PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    if (Cmd.Name == PhdrName)
      return I;
    ++I;
  }
  error("section header '" + PhdrName + "' is not listed in PHDRS");
  return 0;
}

class elf::ScriptParser : public ScriptParserBase {
  typedef void (ScriptParser::*Handler)();

public:
  ScriptParser(StringRef S, bool B) : ScriptParserBase(S), IsUnderSysroot(B) {}

  void readLinkerScript();
  void readVersionScript();

private:
  void addFile(StringRef Path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readPhdrs();
  void readSearchDir();
  void readSections();
  void readVersion();
  void readVersionScriptCommand();

  SymbolAssignment *readAssignment(StringRef Name);
  std::vector<uint8_t> readFill();
  OutputSectionCommand *readOutputSectionDescription(StringRef OutSec);
  std::vector<uint8_t> readOutputSectionFiller(StringRef Tok);
  std::vector<StringRef> readOutputSectionPhdrs();
  InputSectionDescription *readInputSectionDescription(StringRef Tok);
  Regex readFilePatterns();
  void readSectionExcludes(InputSectionDescription *Cmd);
  InputSectionDescription *readInputSectionRules(StringRef FilePattern);
  unsigned readPhdrType();
  SortSectionPolicy readSortKind();
  SymbolAssignment *readProvideHidden(bool Provide, bool Hidden);
  SymbolAssignment *readProvideOrAssignment(StringRef Tok, bool MakeAbsolute);
  void readSort();
  Expr readAssert();

  Expr readExpr();
  Expr readExpr1(Expr Lhs, int MinPrec);
  Expr readPrimary();
  Expr readTernary(Expr Cond);
  Expr readParenExpr();

  // For parsing version script.
  void readExtern(std::vector<SymbolVersion> *Globals);
  void readVersionDeclaration(StringRef VerStr);
  void readGlobal(StringRef VerStr);
  void readLocal();

  ScriptConfiguration &Opt = *ScriptConfig;
  StringSaver Saver = {ScriptConfig->Alloc};
  bool IsUnderSysroot;
};

void ScriptParser::readVersionScript() {
  readVersionScriptCommand();
  if (!atEOF())
    setError("EOF expected, but got " + next());
}

void ScriptParser::readVersionScriptCommand() {
  if (skip("{")) {
    readVersionDeclaration("");
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
    } else if (SymbolAssignment *Cmd = readProvideOrAssignment(Tok, true)) {
      Opt.Commands.emplace_back(Cmd);
    } else {
      setError("unknown directive: " + Tok);
    }
  }
}

void ScriptParser::addFile(StringRef S) {
  if (IsUnderSysroot && S.startswith("/")) {
    SmallString<128> Path;
    (Config->Sysroot + S).toStringRef(Path);
    if (sys::fs::exists(Path)) {
      Driver->addFile(Saver.save(Path.str()));
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
    std::string Path = findFromSearchPaths(S);
    if (Path.empty())
      setError("unable to find " + S);
    else
      Driver->addFile(Saver.save(Path));
  }
}

void ScriptParser::readAsNeeded() {
  expect("(");
  bool Orig = Config->AsNeeded;
  Config->AsNeeded = true;
  while (!Error && !skip(")"))
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
  while (!Error && !skip(")"))
    Config->Undefined.push_back(next());
}

void ScriptParser::readGroup() {
  expect("(");
  while (!Error && !skip(")")) {
    StringRef Tok = next();
    if (Tok == "AS_NEEDED")
      readAsNeeded();
    else
      addFile(unquote(Tok));
  }
}

void ScriptParser::readInclude() {
  StringRef Tok = next();
  auto MBOrErr = MemoryBuffer::getFile(unquote(Tok));
  if (!MBOrErr) {
    setError("cannot open " + Tok);
    return;
  }
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  StringRef S = Saver.save(MB->getMemBufferRef().getBuffer());
  std::vector<StringRef> V = tokenize(S);
  Tokens.insert(Tokens.begin() + Pos, V.begin(), V.end());
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
  // Error checking only for now.
  expect("(");
  next();
  expect(")");
}

void ScriptParser::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  next();
  StringRef Tok = next();
  if (Tok == ")")
    return;
  if (Tok != ",") {
    setError("unexpected token: " + Tok);
    return;
  }
  next();
  expect(",");
  next();
  expect(")");
}

void ScriptParser::readPhdrs() {
  expect("{");
  while (!Error && !skip("}")) {
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
  expect("{");
  while (!Error && !skip("}")) {
    StringRef Tok = next();
    BaseCommand *Cmd = readProvideOrAssignment(Tok, true);
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
      .Case("*", 4)
      .Case("/", 4)
      .Case("+", 3)
      .Case("-", 3)
      .Case("<", 2)
      .Case(">", 2)
      .Case(">=", 2)
      .Case("<=", 2)
      .Case("==", 2)
      .Case("!=", 2)
      .Case("&", 1)
      .Case("|", 1)
      .Default(-1);
}

Regex ScriptParser::readFilePatterns() {
  std::vector<StringRef> V;
  while (!Error && !skip(")"))
    V.push_back(next());
  return compileGlobPatterns(V);
}

SortSectionPolicy ScriptParser::readSortKind() {
  if (skip("SORT") || skip("SORT_BY_NAME"))
    return SortSectionPolicy::Name;
  if (skip("SORT_BY_ALIGNMENT"))
    return SortSectionPolicy::Alignment;
  if (skip("SORT_BY_INIT_PRIORITY"))
    return SortSectionPolicy::Priority;
  if (skip("SORT_NONE"))
    return SortSectionPolicy::None;
  return SortSectionPolicy::Default;
}

static void selectSortKind(InputSectionDescription *Cmd) {
  if (Cmd->SortOuter == SortSectionPolicy::None) {
    Cmd->SortOuter = SortSectionPolicy::Default;
    return;
  }

  if (Cmd->SortOuter != SortSectionPolicy::Default) {
    // If the section sorting command in linker script is nested, the command
    // line option will be ignored.
    if (Cmd->SortInner != SortSectionPolicy::Default)
      return;
    // If the section sorting command in linker script isn't nested, the
    // command line option will make the section sorting command to be treated
    // as nested sorting command.
    Cmd->SortInner = Config->SortSection;
    return;
  }
  // If sorting rule not specified, use command line option.
  Cmd->SortOuter = Config->SortSection;
}

// Method reads a list of sequence of excluded files and section globs given in
// a following form: ((EXCLUDE_FILE(file_pattern+))? section_pattern+)+
// Example: *(.foo.1 EXCLUDE_FILE (*a.o) .foo.2 EXCLUDE_FILE (*b.o) .foo.3)
void ScriptParser::readSectionExcludes(InputSectionDescription *Cmd) {
  Regex ExcludeFileRe;
  std::vector<StringRef> V;

  while (!Error) {
    if (skip(")")) {
      Cmd->SectionPatterns.push_back(
          {std::move(ExcludeFileRe), compileGlobPatterns(V)});
      return;
    }

    if (skip("EXCLUDE_FILE")) {
      if (!V.empty()) {
        Cmd->SectionPatterns.push_back(
            {std::move(ExcludeFileRe), compileGlobPatterns(V)});
        V.clear();
      }

      expect("(");
      ExcludeFileRe = readFilePatterns();
      continue;
    }

    V.push_back(next());
  }
}

InputSectionDescription *
ScriptParser::readInputSectionRules(StringRef FilePattern) {
  auto *Cmd = new InputSectionDescription(FilePattern);
  expect("(");

  // Read SORT().
  SortSectionPolicy K1 = readSortKind();
  if (K1 != SortSectionPolicy::Default) {
    Cmd->SortOuter = K1;
    expect("(");
    SortSectionPolicy K2 = readSortKind();
    if (K2 != SortSectionPolicy::Default) {
      Cmd->SortInner = K2;
      expect("(");
      Cmd->SectionPatterns.push_back({Regex(), readFilePatterns()});
      expect(")");
    } else {
      Cmd->SectionPatterns.push_back({Regex(), readFilePatterns()});
    }
    expect(")");
    selectSortKind(Cmd);
    return Cmd;
  }

  selectSortKind(Cmd);
  readSectionExcludes(Cmd);
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
    for (SectionPattern &Pat : Cmd->SectionPatterns)
      Opt.KeptSections.push_back(&Pat.SectionRe);
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
std::vector<uint8_t> ScriptParser::readFill() {
  expect("(");
  std::vector<uint8_t> V = readOutputSectionFiller(next());
  expect(")");
  expect(";");
  return V;
}

OutputSectionCommand *
ScriptParser::readOutputSectionDescription(StringRef OutSec) {
  OutputSectionCommand *Cmd = new OutputSectionCommand(OutSec);

  // Read an address expression.
  // https://sourceware.org/binutils/docs/ld/Output-Section-Address.html#Output-Section-Address
  if (peek() != ":")
    Cmd->AddrExpr = readExpr();

  expect(":");

  if (skip("AT"))
    Cmd->LmaExpr = readParenExpr();
  if (skip("ALIGN"))
    Cmd->AlignExpr = readParenExpr();
  if (skip("SUBALIGN"))
    Cmd->SubalignExpr = readParenExpr();

  // Parse constraints.
  if (skip("ONLY_IF_RO"))
    Cmd->Constraint = ConstraintKind::ReadOnly;
  if (skip("ONLY_IF_RW"))
    Cmd->Constraint = ConstraintKind::ReadWrite;
  expect("{");

  while (!Error && !skip("}")) {
    StringRef Tok = next();
    if (SymbolAssignment *Assignment = readProvideOrAssignment(Tok, false))
      Cmd->Commands.emplace_back(Assignment);
    else if (Tok == "FILL")
      Cmd->Filler = readFill();
    else if (Tok == "SORT")
      readSort();
    else if (peek() == "(")
      Cmd->Commands.emplace_back(readInputSectionDescription(Tok));
    else
      setError("unknown command " + Tok);
  }
  Cmd->Phdrs = readOutputSectionPhdrs();
  if (peek().startswith("="))
    Cmd->Filler = readOutputSectionFiller(next().drop_front());
  return Cmd;
}

// Read "=<number>" where <number> is an octal/decimal/hexadecimal number.
// https://sourceware.org/binutils/docs/ld/Output-Section-Fill.html
//
// ld.gold is not fully compatible with ld.bfd. ld.bfd handles
// hexstrings as blobs of arbitrary sizes, while ld.gold handles them
// as 32-bit big-endian values. We will do the same as ld.gold does
// because it's simpler than what ld.bfd does.
std::vector<uint8_t> ScriptParser::readOutputSectionFiller(StringRef Tok) {
  uint32_t V;
  if (Tok.getAsInteger(0, V)) {
    setError("invalid filler expression: " + Tok);
    return {};
  }
  return {uint8_t(V >> 24), uint8_t(V >> 16), uint8_t(V >> 8), uint8_t(V)};
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

SymbolAssignment *ScriptParser::readProvideOrAssignment(StringRef Tok,
                                                        bool MakeAbsolute) {
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
  if (Cmd && MakeAbsolute)
    Cmd->IsAbsolute = true;
  return Cmd;
}

static uint64_t getSymbolValue(StringRef S, uint64_t Dot) {
  if (S == ".")
    return Dot;
  return ScriptBase->getSymbolValue(S);
}

SymbolAssignment *ScriptParser::readAssignment(StringRef Name) {
  StringRef Op = next();
  bool IsAbsolute = false;
  Expr E;
  assert(Op == "=" || Op == "+=");
  if (skip("ABSOLUTE")) {
    E = readParenExpr();
    IsAbsolute = true;
  } else {
    E = readExpr();
  }
  if (Op == "+=")
    E = [=](uint64_t Dot) { return getSymbolValue(Name, Dot) + E(Dot); };
  return new SymbolAssignment(Name, E, IsAbsolute);
}

// This is an operator-precedence parser to parse a linker
// script expression.
Expr ScriptParser::readExpr() { return readExpr1(readPrimary(), 0); }

static Expr combine(StringRef Op, Expr L, Expr R) {
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
    return [=](uint64_t Dot) { return L(Dot) + R(Dot); };
  if (Op == "-")
    return [=](uint64_t Dot) { return L(Dot) - R(Dot); };
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
    StringRef Op1 = peek();
    if (Op1 == "?")
      return readTernary(Lhs);
    if (precedence(Op1) < MinPrec)
      break;
    next();
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
    return Target->MaxPageSize;
  error("unknown constant: " + S);
  return 0;
}

// Parses Tok as an integer. Returns true if successful.
// It recognizes hexadecimal (prefixed with "0x" or suffixed with "H")
// and decimal numbers. Decimal numbers may have "K" (kilo) or
// "M" (mega) prefixes.
static bool readInteger(StringRef Tok, uint64_t &Result) {
  if (Tok.startswith("-")) {
    if (!readInteger(Tok.substr(1), Result))
      return false;
    Result = -Result;
    return true;
  }
  if (Tok.startswith_lower("0x"))
    return !Tok.substr(2).getAsInteger(16, Result);
  if (Tok.endswith_lower("H"))
    return !Tok.drop_back().getAsInteger(16, Result);

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

Expr ScriptParser::readPrimary() {
  if (peek() == "(")
    return readParenExpr();

  StringRef Tok = next();

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
    expect("(");
    StringRef Name = next();
    expect(")");
    return
        [=](uint64_t Dot) { return ScriptBase->getOutputSectionAddress(Name); };
  }
  if (Tok == "ASSERT")
    return readAssert();
  if (Tok == "ALIGN") {
    Expr E = readParenExpr();
    return [=](uint64_t Dot) { return alignTo(Dot, E(Dot)); };
  }
  if (Tok == "CONSTANT") {
    expect("(");
    StringRef Tok = next();
    expect(")");
    return [=](uint64_t Dot) { return getConstant(Tok); };
  }
  if (Tok == "SEGMENT_START") {
    expect("(");
    next();
    expect(",");
    uint64_t Val;
    if (next().getAsInteger(0, Val))
      setError("integer expected");
    expect(")");
    return [=](uint64_t Dot) { return Val; };
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
    expect("(");
    StringRef Name = next();
    expect(")");
    return [=](uint64_t Dot) { return ScriptBase->getOutputSectionSize(Name); };
  }
  if (Tok == "ALIGNOF") {
    expect("(");
    StringRef Name = next();
    expect(")");
    return
        [=](uint64_t Dot) { return ScriptBase->getOutputSectionAlign(Name); };
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
  return [=](uint64_t Dot) { return getSymbolValue(Tok, Dot); };
}

Expr ScriptParser::readTernary(Expr Cond) {
  next();
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
    Tok = (Tok.size() == 1) ? next() : Tok.substr(1);
    if (Tok.empty()) {
      setError("section header name is empty");
      break;
    }
    Phdrs.push_back(Tok);
  }
  return Phdrs;
}

unsigned ScriptParser::readPhdrType() {
  StringRef Tok = next();
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
                     .Default(-1);

  if (Ret == (unsigned)-1) {
    setError("invalid program header type: " + Tok);
    return PT_NULL;
  }
  return Ret;
}

void ScriptParser::readVersionDeclaration(StringRef VerStr) {
  // Identifiers start at 2 because 0 and 1 are reserved
  // for VER_NDX_LOCAL and VER_NDX_GLOBAL constants.
  size_t VersionId = Config->VersionDefinitions.size() + 2;
  Config->VersionDefinitions.push_back({VerStr, VersionId});

  if (skip("global:") || peek() != "local:")
    readGlobal(VerStr);
  if (skip("local:"))
    readLocal();
  expect("}");

  // Each version may have a parent version. For example, "Ver2" defined as
  // "Ver2 { global: foo; local: *; } Ver1;" has "Ver1" as a parent. This
  // version hierarchy is, probably against your instinct, purely for human; the
  // runtime doesn't care about them at all. In LLD, we simply skip the token.
  if (!VerStr.empty() && peek() != ";")
    next();
  expect(";");
}

void ScriptParser::readLocal() {
  Config->DefaultSymbolVersion = VER_NDX_LOCAL;
  expect("*");
  expect(";");
}

void ScriptParser::readExtern(std::vector<SymbolVersion> *Globals) {
  expect("\"C++\"");
  expect("{");

  for (;;) {
    if (peek() == "}" || Error)
      break;
    bool HasWildcard = !peek().startswith("\"") && hasWildcard(peek());
    Globals->push_back({unquote(next()), true, HasWildcard});
    expect(";");
  }

  expect("}");
  expect(";");
}

void ScriptParser::readGlobal(StringRef VerStr) {
  std::vector<SymbolVersion> *Globals;
  if (VerStr.empty())
    Globals = &Config->VersionScriptGlobals;
  else
    Globals = &Config->VersionDefinitions.back().Globals;

  for (;;) {
    if (skip("extern"))
      readExtern(Globals);

    StringRef Cur = peek();
    if (Cur == "}" || Cur == "local:" || Error)
      return;
    next();
    Globals->push_back({unquote(Cur), false, hasWildcard(Cur)});
    expect(";");
  }
}

static bool isUnderSysroot(StringRef Path) {
  if (Config->Sysroot == "")
    return false;
  for (; !Path.empty(); Path = sys::path::parent_path(Path))
    if (sys::fs::equivalent(Config->Sysroot, Path))
      return true;
  return false;
}

void elf::readLinkerScript(MemoryBufferRef MB) {
  StringRef Path = MB.getBufferIdentifier();
  ScriptParser(MB.getBuffer(), isUnderSysroot(Path)).readLinkerScript();
}

void elf::readVersionScript(MemoryBufferRef MB) {
  ScriptParser(MB.getBuffer(), false).readVersionScript();
}

template class elf::LinkerScript<ELF32LE>;
template class elf::LinkerScript<ELF32BE>;
template class elf::LinkerScript<ELF64LE>;
template class elf::LinkerScript<ELF64BE>;
