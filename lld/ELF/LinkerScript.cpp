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

ScriptConfiguration *elf::ScriptConfig;

template <class ELFT>
static void addRegular(SymbolAssignment *Cmd) {
  Symbol *Sym = Symtab<ELFT>::X->addRegular(Cmd->Name, STB_GLOBAL, STV_DEFAULT);
  Sym->Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  Cmd->Sym = Sym->body();
}

template <class ELFT> static void addSynthetic(SymbolAssignment *Cmd) {
  Symbol *Sym = Symtab<ELFT>::X->addSynthetic(Cmd->Name, nullptr, 0);
  Sym->Visibility = Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT;
  Cmd->Sym = Sym->body();
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
  for (StringRef Pat : Opt.KeptSections)
    if (globMatch(Pat, S->getSectionName()))
      return true;
  return false;
}

static bool match(ArrayRef<StringRef> Patterns, StringRef S) {
  for (StringRef Pat : Patterns)
    if (globMatch(Pat, S))
      return true;
  return false;
}

static bool fileMatches(const InputSectionDescription *Desc,
                        StringRef Filename) {
  if (!globMatch(Desc->FilePattern, Filename))
    return false;
  return Desc->ExcludedFiles.empty() || !match(Desc->ExcludedFiles, Filename);
}

// Returns input sections filtered by given glob patterns.
template <class ELFT>
std::vector<InputSectionBase<ELFT> *>
LinkerScript<ELFT>::getInputSections(const InputSectionDescription *I) {
  ArrayRef<StringRef> Patterns = I->SectionPatterns;
  std::vector<InputSectionBase<ELFT> *> Ret;
  for (const std::unique_ptr<ObjectFile<ELFT>> &F :
       Symtab<ELFT>::X->getObjectFiles()) {
    if (fileMatches(I, sys::path::filename(F->getName())))
      for (InputSectionBase<ELFT> *S : F->getSections())
        if (!isDiscarded(S) && !S->OutSec &&
            match(Patterns, S->getSectionName()))
          Ret.push_back(S);
  }

  if (llvm::find(Patterns, "COMMON") != Patterns.end())
    Ret.push_back(CommonInputSection<ELFT>::X);

  return Ret;
}

// You can define new symbols using linker scripts. For example,
// ".text { abc.o(.text); foo = .; def.o(.text); }" defines symbol
// foo just after abc.o's text section contents. This class is to
// handle such symbol definitions.
//
// In order to handle scripts like the above one, we want to
// keep symbol definitions in output sections. Because output sections
// can contain only input sections, we wrap symbol definitions
// with dummy input sections. This class serves that purpose.
template <class ELFT>
class elf::LayoutInputSection : public InputSectionBase<ELFT> {
public:
  explicit LayoutInputSection(SymbolAssignment *Cmd);
  static bool classof(const InputSectionBase<ELFT> *S);
  SymbolAssignment *Cmd;

private:
  typename ELFT::Shdr Hdr;
};

template <class ELFT>
static InputSectionBase<ELFT> *
getNonLayoutSection(std::vector<InputSectionBase<ELFT> *> &Vec) {
  for (InputSectionBase<ELFT> *S : Vec)
    if (!isa<LayoutInputSection<ELFT>>(S))
      return S;
  return nullptr;
}

template <class T> static T *zero(T *Val) {
  memset(Val, 0, sizeof(*Val));
  return Val;
}

template <class ELFT>
LayoutInputSection<ELFT>::LayoutInputSection(SymbolAssignment *Cmd)
    : InputSectionBase<ELFT>(nullptr, zero(&Hdr),
                             InputSectionBase<ELFT>::Layout),
      Cmd(Cmd) {
  this->Live = true;
  Hdr.sh_type = SHT_NOBITS;
}

template <class ELFT>
bool LayoutInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::Layout;
}

template <class ELFT>
static bool compareName(InputSectionBase<ELFT> *A, InputSectionBase<ELFT> *B) {
  return A->getSectionName() < B->getSectionName();
}

template <class ELFT>
static bool compareAlignment(InputSectionBase<ELFT> *A,
                             InputSectionBase<ELFT> *B) {
  // ">" is not a mistake. Larger alignments are placed before smaller
  // alignments in order to reduce the amount of padding necessary.
  // This is compatible with GNU.
  return A->Alignment > B->Alignment;
}

template <class ELFT>
static std::function<bool(InputSectionBase<ELFT> *, InputSectionBase<ELFT> *)>
getComparator(SortKind K) {
  if (K == SortByName)
    return compareName<ELFT>;
  return compareAlignment<ELFT>;
}

template <class ELFT>
void LinkerScript<ELFT>::discard(OutputSectionCommand &Cmd) {
  for (const std::unique_ptr<BaseCommand> &Base : Cmd.Commands) {
    if (auto *Cmd = dyn_cast<InputSectionDescription>(Base.get())) {
      for (InputSectionBase<ELFT> *S : getInputSections(Cmd)) {
        S->Live = false;
        reportDiscarded(S);
      }
    }
  }
}

static bool checkConstraint(uint64_t Flags, ConstraintKind Kind) {
  bool RO = (Kind == ConstraintKind::ReadOnly);
  bool RW = (Kind == ConstraintKind::ReadWrite);
  bool Writable = Flags & SHF_WRITE;
  return !((RO && Writable) || (RW && !Writable));
}

template <class ELFT>
static bool matchConstraints(ArrayRef<InputSectionBase<ELFT> *> Sections,
                             ConstraintKind Kind) {
  if (Kind == ConstraintKind::NoConstraint)
    return true;
  return llvm::all_of(Sections, [=](InputSectionBase<ELFT> *Sec) {
    return checkConstraint(Sec->getSectionHdr()->sh_flags, Kind);
  });
}

template <class ELFT>
std::vector<InputSectionBase<ELFT> *>
LinkerScript<ELFT>::createInputSectionList(OutputSectionCommand &OutCmd) {
  std::vector<InputSectionBase<ELFT> *> Ret;

  for (const std::unique_ptr<BaseCommand> &Base : OutCmd.Commands) {
    if (auto *OutCmd = dyn_cast<SymbolAssignment>(Base.get())) {
      if (shouldDefine<ELFT>(OutCmd))
        addSynthetic<ELFT>(OutCmd);
      Ret.push_back(new (LAlloc.Allocate()) LayoutInputSection<ELFT>(OutCmd));
      continue;
    }

    auto *Cmd = cast<InputSectionDescription>(Base.get());
    std::vector<InputSectionBase<ELFT> *> V = getInputSections(Cmd);
    if (!matchConstraints<ELFT>(V, OutCmd.Constraint))
      continue;
    if (Cmd->SortInner)
      std::stable_sort(V.begin(), V.end(), getComparator<ELFT>(Cmd->SortInner));
    if (Cmd->SortOuter)
      std::stable_sort(V.begin(), V.end(), getComparator<ELFT>(Cmd->SortOuter));
    Ret.insert(Ret.end(), V.begin(), V.end());
  }
  return Ret;
}

template <class ELFT>
void LinkerScript<ELFT>::createSections(OutputSectionFactory<ELFT> &Factory) {
  for (const std::unique_ptr<BaseCommand> &Base1 : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base1.get())) {
      if (shouldDefine<ELFT>(Cmd))
        addRegular<ELFT>(Cmd);
      continue;
    }

    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base1.get())) {
      if (Cmd->Name == "/DISCARD/") {
        discard(*Cmd);
        continue;
      }

      std::vector<InputSectionBase<ELFT> *> V = createInputSectionList(*Cmd);
      InputSectionBase<ELFT> *Head = getNonLayoutSection<ELFT>(V);
      if (!Head)
        continue;

      OutputSectionBase<ELFT> *OutSec;
      bool IsNew;
      std::tie(OutSec, IsNew) = Factory.create(Head, Cmd->Name);
      if (IsNew)
        OutputSections->push_back(OutSec);
      for (InputSectionBase<ELFT> *Sec : V)
        OutSec->addSection(Sec);
    }
  }

  // Add orphan sections.
  for (const std::unique_ptr<ObjectFile<ELFT>> &F :
       Symtab<ELFT>::X->getObjectFiles()) {
    for (InputSectionBase<ELFT> *S : F->getSections()) {
      if (isDiscarded(S) || S->OutSec)
        continue;
      OutputSectionBase<ELFT> *OutSec;
      bool IsNew;
      std::tie(OutSec, IsNew) = Factory.create(S, getOutputSectionName(S));
      if (IsNew)
        OutputSections->push_back(OutSec);
      OutSec->addSection(S);
    }
  }
}

template <class ELFT> void assignOffsets(OutputSectionBase<ELFT> *Sec) {
  auto *OutSec = dyn_cast<OutputSection<ELFT>>(Sec);
  if (!OutSec) {
    Sec->assignOffsets();
    return;
  }

  typedef typename ELFT::uint uintX_t;
  uintX_t Off = 0;

  for (InputSection<ELFT> *I : OutSec->Sections) {
    if (auto *L = dyn_cast<LayoutInputSection<ELFT>>(I)) {
      uintX_t Value = L->Cmd->Expression(Sec->getVA() + Off) - Sec->getVA();
      if (L->Cmd->Name == ".") {
        Off = Value;
      } else if (auto *Sym =
                     cast_or_null<DefinedSynthetic<ELFT>>(L->Cmd->Sym)) {
        // shouldDefine could have returned false, so we need to check Sym,
        // for non-null value.
        Sym->Section = OutSec;
        Sym->Value = Value;
      }
    } else {
      Off = alignTo(Off, I->Alignment);
      I->OutSecOff = Off;
      Off += I->getSize();
    }
    // Update section size inside for-loop, so that SIZEOF
    // works correctly in the case below:
    // .foo { *(.aaa) a = SIZEOF(.foo); *(.bbb) }
    Sec->setSize(Off);
  }
}

template <class ELFT>
static OutputSectionBase<ELFT> *
findSection(OutputSectionCommand &Cmd,
            ArrayRef<OutputSectionBase<ELFT> *> Sections) {
  for (OutputSectionBase<ELFT> *Sec : Sections) {
    if (Sec->getName() != Cmd.Name)
      continue;
    if (checkConstraint(Sec->getFlags(), Cmd.Constraint))
      return Sec;
  }
  return nullptr;
}

template <class ELFT> void LinkerScript<ELFT>::assignAddresses() {
  // Orphan sections are sections present in the input files which
  // are not explicitly placed into the output file by the linker script.
  // We place orphan sections at end of file.
  // Other linkers places them using some heuristics as described in
  // https://sourceware.org/binutils/docs/ld/Orphan-Sections.html#Orphan-Sections.
  for (OutputSectionBase<ELFT> *Sec : *OutputSections) {
    StringRef Name = Sec->getName();
    if (getSectionIndex(Name) == INT_MAX)
      Opt.Commands.push_back(llvm::make_unique<OutputSectionCommand>(Name));
  }

  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = getHeaderSize();
  uintX_t MinVA = std::numeric_limits<uintX_t>::max();
  uintX_t ThreadBssOffset = 0;

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
    OutputSectionBase<ELFT> *Sec = findSection<ELFT>(*Cmd, *OutputSections);
    if (!Sec)
      continue;

    if (Cmd->AddrExpr)
      Dot = Cmd->AddrExpr(Dot);

    if (Cmd->AlignExpr)
      Sec->updateAlignment(Cmd->AlignExpr(Dot));

    if ((Sec->getFlags() & SHF_TLS) && Sec->getType() == SHT_NOBITS) {
      uintX_t TVA = Dot + ThreadBssOffset;
      TVA = alignTo(TVA, Sec->getAlignment());
      Sec->setVA(TVA);
      assignOffsets(Sec);
      ThreadBssOffset = TVA - Dot + Sec->getSize();
      continue;
    }

    if (!(Sec->getFlags() & SHF_ALLOC)) {
      Sec->assignOffsets();
      continue;
    }

    Dot = alignTo(Dot, Sec->getAlignment());
    Sec->setVA(Dot);
    assignOffsets(Sec);
    MinVA = std::min(MinVA, Dot);
    Dot += Sec->getSize();
  }

  // ELF and Program headers need to be right before the first section in
  // memory. Set their addresses accordingly.
  MinVA = alignDown(MinVA - Out<ELFT>::ElfHeader->getSize() -
                        Out<ELFT>::ProgramHeaders->getSize(),
                    Target->PageSize);
  Out<ELFT>::ElfHeader->setVA(MinVA);
  Out<ELFT>::ProgramHeaders->setVA(Out<ELFT>::ElfHeader->getSize() + MinVA);
}

template <class ELFT>
std::vector<PhdrEntry<ELFT>> LinkerScript<ELFT>::createPhdrs() {
  ArrayRef<OutputSectionBase<ELFT> *> Sections = *OutputSections;
  std::vector<PhdrEntry<ELFT>> Ret;

  for (const PhdrsCommand &Cmd : Opt.PhdrsCommands) {
    Ret.emplace_back(Cmd.Type, Cmd.Flags == UINT_MAX ? PF_R : Cmd.Flags);
    PhdrEntry<ELFT> &Phdr = Ret.back();

    if (Cmd.HasFilehdr)
      Phdr.add(Out<ELFT>::ElfHeader);
    if (Cmd.HasPhdrs)
      Phdr.add(Out<ELFT>::ProgramHeaders);

    switch (Cmd.Type) {
    case PT_INTERP:
      if (Out<ELFT>::Interp)
        Phdr.add(Out<ELFT>::Interp);
      break;
    case PT_DYNAMIC:
      if (Out<ELFT>::DynSymTab) {
        Phdr.H.p_flags = Out<ELFT>::Dynamic->getPhdrFlags();
        Phdr.add(Out<ELFT>::Dynamic);
      }
      break;
    case PT_GNU_EH_FRAME:
      if (!Out<ELFT>::EhFrame->empty() && Out<ELFT>::EhFrameHdr) {
        Phdr.H.p_flags = Out<ELFT>::EhFrameHdr->getPhdrFlags();
        Phdr.add(Out<ELFT>::EhFrameHdr);
      }
      break;
    }
  }

  PhdrEntry<ELFT> *Load = nullptr;
  uintX_t Flags = PF_R;
  for (OutputSectionBase<ELFT> *Sec : Sections) {
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
typename ELFT::uint LinkerScript<ELFT>::getOutputSectionSize(StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : *OutputSections)
    if (Sec->getName() == Name)
      return Sec->getSize();
  error("undefined section " + Name);
  return 0;
}

template <class ELFT>
typename ELFT::uint LinkerScript<ELFT>::getHeaderSize() {
  return Out<ELFT>::ElfHeader->getSize() + Out<ELFT>::ProgramHeaders->getSize();
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

  void run();

private:
  void addFile(StringRef Path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readNothing() {}
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readPhdrs();
  void readSearchDir();
  void readSections();

  SymbolAssignment *readAssignment(StringRef Name);
  OutputSectionCommand *readOutputSectionDescription(StringRef OutSec);
  std::vector<uint8_t> readOutputSectionFiller();
  std::vector<StringRef> readOutputSectionPhdrs();
  InputSectionDescription *readInputSectionDescription();
  std::vector<StringRef> readInputFilePatterns();
  InputSectionDescription *readInputSectionRules();
  unsigned readPhdrType();
  SortKind readSortKind();
  SymbolAssignment *readProvideHidden(bool Provide, bool Hidden);
  SymbolAssignment *readProvideOrAssignment(StringRef Tok);
  void readSort();
  Expr readAssert();

  Expr readExpr();
  Expr readExpr1(Expr Lhs, int MinPrec);
  Expr readPrimary();
  Expr readTernary(Expr Cond);
  Expr readParenExpr();

  const static StringMap<Handler> Cmd;
  ScriptConfiguration &Opt = *ScriptConfig;
  StringSaver Saver = {ScriptConfig->Alloc};
  bool IsUnderSysroot;
};

const StringMap<elf::ScriptParser::Handler> elf::ScriptParser::Cmd = {
    {"ENTRY", &ScriptParser::readEntry},
    {"EXTERN", &ScriptParser::readExtern},
    {"GROUP", &ScriptParser::readGroup},
    {"INCLUDE", &ScriptParser::readInclude},
    {"INPUT", &ScriptParser::readGroup},
    {"OUTPUT", &ScriptParser::readOutput},
    {"OUTPUT_ARCH", &ScriptParser::readOutputArch},
    {"OUTPUT_FORMAT", &ScriptParser::readOutputFormat},
    {"PHDRS", &ScriptParser::readPhdrs},
    {"SEARCH_DIR", &ScriptParser::readSearchDir},
    {"SECTIONS", &ScriptParser::readSections},
    {";", &ScriptParser::readNothing}};

void ScriptParser::run() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Handler Fn = Cmd.lookup(Tok))
      (this->*Fn)();
    else
      setError("unknown directive: " + Tok);
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
    addFile(next());
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
      addFile(Tok);
  }
}

void ScriptParser::readInclude() {
  StringRef Tok = next();
  auto MBOrErr = MemoryBuffer::getFile(Tok);
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
    Config->OutputFile = Tok;
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
    Opt.PhdrsCommands.push_back({Tok, PT_NULL, false, false, UINT_MAX});
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
  Config->SearchPaths.push_back(next());
  expect(")");
}

void ScriptParser::readSections() {
  Opt.HasContents = true;
  expect("{");
  while (!Error && !skip("}")) {
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
      .Default(-1);
}

std::vector<StringRef> ScriptParser::readInputFilePatterns() {
  std::vector<StringRef> V;
  while (!Error && !skip(")"))
    V.push_back(next());
  return V;
}

SortKind ScriptParser::readSortKind() {
  if (skip("SORT") || skip("SORT_BY_NAME"))
    return SortByName;
  if (skip("SORT_BY_ALIGNMENT"))
    return SortByAlignment;
  return SortNone;
}

InputSectionDescription *ScriptParser::readInputSectionRules() {
  auto *Cmd = new InputSectionDescription;
  Cmd->FilePattern = next();
  expect("(");

  // Read EXCLUDE_FILE().
  if (skip("EXCLUDE_FILE")) {
    expect("(");
    while (!Error && !skip(")"))
      Cmd->ExcludedFiles.push_back(next());
  }

  // Read SORT().
  if (SortKind K1 = readSortKind()) {
    Cmd->SortOuter = K1;
    expect("(");
    if (SortKind K2 = readSortKind()) {
      Cmd->SortInner = K2;
      expect("(");
      Cmd->SectionPatterns = readInputFilePatterns();
      expect(")");
    } else {
      Cmd->SectionPatterns = readInputFilePatterns();
    }
    expect(")");
    return Cmd;
  }

  Cmd->SectionPatterns = readInputFilePatterns();
  return Cmd;
}

InputSectionDescription *ScriptParser::readInputSectionDescription() {
  // Input section wildcard can be surrounded by KEEP.
  // https://sourceware.org/binutils/docs/ld/Input-Section-Keep.html#Input-Section-Keep
  if (skip("KEEP")) {
    expect("(");
    InputSectionDescription *Cmd = readInputSectionRules();
    expect(")");
    Opt.KeptSections.insert(Opt.KeptSections.end(),
                            Cmd->SectionPatterns.begin(),
                            Cmd->SectionPatterns.end());
    return Cmd;
  }
  return readInputSectionRules();
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
  StringRef Msg = next();
  expect(")");
  return [=](uint64_t Dot) {
    uint64_t V = E(Dot);
    if (!V)
      error(Msg);
    return V;
  };
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

  // Parse constraints.
  if (skip("ONLY_IF_RO"))
    Cmd->Constraint = ConstraintKind::ReadOnly;
  if (skip("ONLY_IF_RW"))
    Cmd->Constraint = ConstraintKind::ReadWrite;
  expect("{");

  while (!Error && !skip("}")) {
    if (peek().startswith("*") || peek() == "KEEP") {
      Cmd->Commands.emplace_back(readInputSectionDescription());
      continue;
    }

    StringRef Tok = next();
    if (SymbolAssignment *Assignment = readProvideOrAssignment(Tok))
      Cmd->Commands.emplace_back(Assignment);
    else if (Tok == "SORT")
      readSort();
    else
      setError("unknown command " + Tok);
  }
  Cmd->Phdrs = readOutputSectionPhdrs();
  Cmd->Filler = readOutputSectionFiller();
  return Cmd;
}

std::vector<uint8_t> ScriptParser::readOutputSectionFiller() {
  StringRef Tok = peek();
  if (!Tok.startswith("="))
    return {};
  next();

  // Read a hexstring of arbitrary length.
  if (Tok.startswith("=0x"))
    return parseHex(Tok.substr(3));

  // Read a decimal or octal value as a big-endian 32 bit value.
  // Why do this? I don't know, but that's what gold does.
  uint32_t V;
  if (Tok.substr(1).getAsInteger(0, V)) {
    setError("invalid filler expression: " + Tok);
    return {};
  }
  return { uint8_t(V >> 24), uint8_t(V >> 16), uint8_t(V >> 8), uint8_t(V) };
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

static uint64_t getSymbolValue(StringRef S, uint64_t Dot) {
  if (S == ".")
    return Dot;

  switch (Config->EKind) {
  case ELF32LEKind:
    if (SymbolBody *B = Symtab<ELF32LE>::X->find(S))
      return B->getVA<ELF32LE>();
    break;
  case ELF32BEKind:
    if (SymbolBody *B = Symtab<ELF32BE>::X->find(S))
      return B->getVA<ELF32BE>();
    break;
  case ELF64LEKind:
    if (SymbolBody *B = Symtab<ELF64LE>::X->find(S))
      return B->getVA<ELF64LE>();
    break;
  case ELF64BEKind:
    if (SymbolBody *B = Symtab<ELF64BE>::X->find(S))
      return B->getVA<ELF64BE>();
    break;
  default:
    llvm_unreachable("unsupported target");
  }
  error("symbol not found: " + S);
  return 0;
}

static uint64_t getSectionSize(StringRef Name) {
  switch (Config->EKind) {
  case ELF32LEKind:
    return Script<ELF32LE>::X->getOutputSectionSize(Name);
  case ELF32BEKind:
    return Script<ELF32BE>::X->getOutputSectionSize(Name);
  case ELF64LEKind:
    return Script<ELF64LE>::X->getOutputSectionSize(Name);
  case ELF64BEKind:
    return Script<ELF64BE>::X->getOutputSectionSize(Name);
  default:
    llvm_unreachable("unsupported target");
  }
}

static uint64_t getHeaderSize() {
  switch (Config->EKind) {
  case ELF32LEKind:
    return Script<ELF32LE>::X->getHeaderSize();
  case ELF32BEKind:
    return Script<ELF32BE>::X->getHeaderSize();
  case ELF64LEKind:
    return Script<ELF64LE>::X->getHeaderSize();
  case ELF64BEKind:
    return Script<ELF64BE>::X->getHeaderSize();
  default:
    llvm_unreachable("unsupported target");
  }
}

SymbolAssignment *ScriptParser::readAssignment(StringRef Name) {
  StringRef Op = next();
  assert(Op == "=" || Op == "+=");
  Expr E = readExpr();
  if (Op == "+=")
    E = [=](uint64_t Dot) { return getSymbolValue(Name, Dot) + E(Dot); };
  return new SymbolAssignment(Name, E);
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

Expr ScriptParser::readPrimary() {
  if (peek() == "(")
    return readParenExpr();

  StringRef Tok = next();

  // Built-in functions are parsed here.
  // https://sourceware.org/binutils/docs/ld/Builtin-Functions.html.
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
    next().getAsInteger(0, Val);
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
    next();
    expect(",");
    readExpr();
    expect(")");
    return [](uint64_t Dot) { return alignTo(Dot, Target->PageSize); };
  }
  if (Tok == "SIZEOF") {
    expect("(");
    StringRef Name = next();
    expect(")");
    return [=](uint64_t Dot) { return getSectionSize(Name); };
  }
  if (Tok == "SIZEOF_HEADERS")
    return [=](uint64_t Dot) { return getHeaderSize(); };

  // Parse a symbol name or a number literal.
  uint64_t V = 0;
  if (Tok.getAsInteger(0, V)) {
    if (Tok != "." && !isValidCIdentifier(Tok))
      setError("malformed number: " + Tok);
    return [=](uint64_t Dot) { return getSymbolValue(Tok, Dot); };
  }
  return [=](uint64_t Dot) { return V; };
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

static bool isUnderSysroot(StringRef Path) {
  if (Config->Sysroot == "")
    return false;
  for (; !Path.empty(); Path = sys::path::parent_path(Path))
    if (sys::fs::equivalent(Config->Sysroot, Path))
      return true;
  return false;
}

// Entry point.
void elf::readLinkerScript(MemoryBufferRef MB) {
  StringRef Path = MB.getBufferIdentifier();
  ScriptParser(MB.getBuffer(), isUnderSysroot(Path)).run();
}

template class elf::LinkerScript<ELF32LE>;
template class elf::LinkerScript<ELF32BE>;
template class elf::LinkerScript<ELF64LE>;
template class elf::LinkerScript<ELF64BE>;
