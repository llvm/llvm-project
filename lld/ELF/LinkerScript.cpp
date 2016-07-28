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

bool SymbolAssignment::classof(const BaseCommand *C) {
  return C->Kind == AssignmentKind;
}

bool OutputSectionCommand::classof(const BaseCommand *C) {
  return C->Kind == OutputSectionKind;
}

bool InputSectionDescription::classof(const BaseCommand *C) {
  return C->Kind == InputSectionKind;
}

template <class ELFT> static bool isDiscarded(InputSectionBase<ELFT> *S) {
  return !S || !S->Live;
}

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

// Create a vector of (<output section name>, <input section name patterns>).
// For example, if a returned vector contains (".text" (".foo.*" ".bar.*")),
// input sections start with ".foo." or ".bar." should be added to
// ".text" section.
template <class ELFT>
std::vector<std::pair<StringRef, const InputSectionDescription *>>
LinkerScript<ELFT>::getSectionMap() {
  std::vector<std::pair<StringRef, const InputSectionDescription *>> Ret;

  for (const std::unique_ptr<BaseCommand> &Base1 : Opt.Commands)
    if (auto *Cmd1 = dyn_cast<OutputSectionCommand>(Base1.get()))
      for (const std::unique_ptr<BaseCommand> &Base2 : Cmd1->Commands)
        if (auto *Cmd2 = dyn_cast<InputSectionDescription>(Base2.get()))
          Ret.emplace_back(Cmd1->Name, Cmd2);

  return Ret;
}

// Returns input sections filtered by given glob patterns.
template <class ELFT>
std::vector<InputSectionBase<ELFT> *>
LinkerScript<ELFT>::getInputSections(const InputSectionDescription *I) {
  ArrayRef<StringRef> Patterns = I->Patterns;
  ArrayRef<StringRef> ExcludedFiles = I->ExcludedFiles;
  std::vector<InputSectionBase<ELFT> *> Ret;
  for (const std::unique_ptr<ObjectFile<ELFT>> &F :
       Symtab<ELFT>::X->getObjectFiles())
    for (InputSectionBase<ELFT> *S : F->getSections())
      if (!isDiscarded(S) && !S->OutSec && match(Patterns, S->getSectionName()))
        if (ExcludedFiles.empty() ||
            !match(ExcludedFiles, sys::path::filename(F->getName())))
          Ret.push_back(S);
  return Ret;
}

template <class ELFT>
std::vector<OutputSectionBase<ELFT> *>
LinkerScript<ELFT>::createSections(OutputSectionFactory<ELFT> &Factory) {
  std::vector<OutputSectionBase<ELFT> *> Ret;

  // Add input section to output section. If there is no output section yet,
  // then create it and add to output section list.
  auto Add = [&](InputSectionBase<ELFT> *C, StringRef Name) {
    OutputSectionBase<ELFT> *Sec;
    bool IsNew;
    std::tie(Sec, IsNew) = Factory.create(C, Name);
    if (IsNew)
      Ret.push_back(Sec);
    Sec->addSection(C);
  };

  for (auto &P : getSectionMap()) {
    StringRef OutputName = P.first;
    const InputSectionDescription *I = P.second;
    for (InputSectionBase<ELFT> *S : getInputSections(I)) {
      if (OutputName == "/DISCARD/") {
        S->Live = false;
        reportDiscarded(S);
        continue;
      }
      Add(S, OutputName);
    }
  }

  // Add all other input sections, which are not listed in script.
  for (const std::unique_ptr<ObjectFile<ELFT>> &F :
       Symtab<ELFT>::X->getObjectFiles())
    for (InputSectionBase<ELFT> *S : F->getSections())
      if (!isDiscarded(S) && !S->OutSec)
        Add(S, getOutputSectionName(S));

  // Remove from the output all the sections which did not meet
  // the optional constraints.
  return filter(Ret);
}

// Process ONLY_IF_RO and ONLY_IF_RW.
template <class ELFT>
std::vector<OutputSectionBase<ELFT> *>
LinkerScript<ELFT>::filter(std::vector<OutputSectionBase<ELFT> *> &Sections) {
  // In this loop, we remove output sections if they don't satisfy
  // requested properties.
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get());
    if (!Cmd || Cmd->Name == "/DISCARD/")
      continue;

    if (Cmd->Constraint == ConstraintKind::NoConstraint)
      continue;

    auto It = llvm::find_if(Sections, [&](OutputSectionBase<ELFT> *S) {
      return S->getName() == Cmd->Name;
    });
    if (It == Sections.end())
      continue;

    OutputSectionBase<ELFT> *Sec = *It;
    bool Writable = (Sec->getFlags() & SHF_WRITE);
    bool RO = (Cmd->Constraint == ConstraintKind::ReadOnly);
    bool RW = (Cmd->Constraint == ConstraintKind::ReadWrite);

    if ((RO && Writable) || (RW && !Writable)) {
      Sections.erase(It);
      continue;
    }
  }
  return Sections;
}

template <class ELFT>
void LinkerScript<ELFT>::dispatchAssignment(SymbolAssignment *Cmd) {
  uint64_t Val = Cmd->Expression(Dot);
  if (Cmd->Name == ".") {
    Dot = Val;
  } else if (!Cmd->Ignore) {
    auto *D = cast<DefinedRegular<ELFT>>(Symtab<ELFT>::X->find(Cmd->Name));
    D->Value = Val;
  }
}

template <class ELFT>
void LinkerScript<ELFT>::assignAddresses(
    ArrayRef<OutputSectionBase<ELFT> *> Sections) {
  // Orphan sections are sections present in the input files which
  // are not explicitly placed into the output file by the linker script.
  // We place orphan sections at end of file.
  // Other linkers places them using some heuristics as described in
  // https://sourceware.org/binutils/docs/ld/Orphan-Sections.html#Orphan-Sections.
  for (OutputSectionBase<ELFT> *Sec : Sections) {
    StringRef Name = Sec->getName();
    if (getSectionIndex(Name) == INT_MAX)
      Opt.Commands.push_back(llvm::make_unique<OutputSectionCommand>(Name));
  }

  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  Dot = Out<ELFT>::ElfHeader->getSize() + Out<ELFT>::ProgramHeaders->getSize();
  uintX_t MinVA = std::numeric_limits<uintX_t>::max();
  uintX_t ThreadBssOffset = 0;

  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    if (auto *Cmd = dyn_cast<SymbolAssignment>(Base.get())) {
      dispatchAssignment(Cmd);
      continue;
    }

    // Find all the sections with required name. There can be more than
    // one section with such name, if the alignment, flags or type
    // attribute differs.
    auto *Cmd = cast<OutputSectionCommand>(Base.get());
    for (OutputSectionBase<ELFT> *Sec : Sections) {
      if (Sec->getName() != Cmd->Name)
        continue;

      if (Cmd->AddrExpr)
        Dot = Cmd->AddrExpr(Dot);

      if (Cmd->AlignExpr)
        Sec->updateAlignment(Cmd->AlignExpr(Dot));

      if ((Sec->getFlags() & SHF_TLS) && Sec->getType() == SHT_NOBITS) {
        uintX_t TVA = Dot + ThreadBssOffset;
        TVA = alignTo(TVA, Sec->getAlignment());
        Sec->setVA(TVA);
        ThreadBssOffset = TVA - Dot + Sec->getSize();
        continue;
      }

      if (Sec->getFlags() & SHF_ALLOC) {
        Dot = alignTo(Dot, Sec->getAlignment());
        Sec->setVA(Dot);
        MinVA = std::min(MinVA, Dot);
        Dot += Sec->getSize();
        continue;
      }
    }
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
std::vector<PhdrEntry<ELFT>>
LinkerScript<ELFT>::createPhdrs(ArrayRef<OutputSectionBase<ELFT> *> Sections) {
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
      if (isOutputDynamic<ELFT>()) {
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

template <class ELFT>
ArrayRef<uint8_t> LinkerScript<ELFT>::getFiller(StringRef Name) {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands)
    if (auto *Cmd = dyn_cast<OutputSectionCommand>(Base.get()))
      if (Cmd->Name == Name)
        return Cmd->Filler;
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

template <class ELFT> void LinkerScript<ELFT>::addScriptedSymbols() {
  for (const std::unique_ptr<BaseCommand> &Base : Opt.Commands) {
    auto *Cmd = dyn_cast<SymbolAssignment>(Base.get());
    if (!Cmd || Cmd->Name == ".")
      continue;

    SymbolBody *B = Symtab<ELFT>::X->find(Cmd->Name);
    // The semantic of PROVIDE is that of introducing a symbol only if
    // it's not defined and there's at least a reference to it.
    if ((!B && !Cmd->Provide) || (B && B->isUndefined()))
      Symtab<ELFT>::X->addAbsolute(Cmd->Name,
                                   Cmd->Hidden ? STV_HIDDEN : STV_DEFAULT);
    else
      // Symbol already exists in symbol table. If it is provided
      // then we can't override its value.
      Cmd->Ignore = Cmd->Provide;
  }
}

template <class ELFT> bool LinkerScript<ELFT>::hasPhdrsCommands() {
  return !Opt.PhdrsCommands.empty();
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
  std::unique_ptr<InputSectionDescription> readFilePattern();
  void readGroup();
  void readKeep(OutputSectionCommand *Cmd);
  void readInclude();
  void readNothing() {}
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readPhdrs();
  void readSearchDir();
  void readSections();

  SymbolAssignment *readAssignment(StringRef Name);
  void readOutputSectionDescription(StringRef OutSec);
  std::vector<StringRef> readOutputSectionPhdrs();
  unsigned readPhdrType();
  void readProvide(bool Hidden);
  void readAlign(OutputSectionCommand *Cmd);

  Expr readExpr();
  Expr readExpr1(Expr Lhs, int MinPrec);
  Expr readPrimary();
  Expr readTernary(Expr Cond);
  Expr combine(StringRef Op, Expr Lhs, Expr Rhs);

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
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ")")
      break;
    addFile(Tok);
  }
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
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    Config->Undefined.push_back(Tok);
  }
}

void ScriptParser::readGroup() {
  expect("(");
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    if (Tok == "AS_NEEDED") {
      readAsNeeded();
      continue;
    }
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
        next().getAsInteger(0, PhdrCmd.Flags);
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
  Opt.DoLayout = true;
  expect("{");
  while (!Error && !skip("}")) {
    StringRef Tok = next();
    if (peek() == "=") {
      readAssignment(Tok);
      expect(";");
    } else if (Tok == "PROVIDE") {
      readProvide(false);
    } else if (Tok == "PROVIDE_HIDDEN") {
      readProvide(true);
    } else {
      readOutputSectionDescription(Tok);
    }
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

std::unique_ptr<InputSectionDescription> ScriptParser::readFilePattern() {
  expect("*");
  expect("(");

  auto InCmd = llvm::make_unique<InputSectionDescription>();

  if (skip("EXCLUDE_FILE")) {
    expect("(");
    while (!Error && !skip(")"))
      InCmd->ExcludedFiles.push_back(next());
    InCmd->Patterns.push_back(next());
    expect(")");
  } else {
    while (!Error && !skip(")"))
      InCmd->Patterns.push_back(next());
  }
  return InCmd;
}

void ScriptParser::readKeep(OutputSectionCommand *Cmd) {
  expect("(");
  std::unique_ptr<InputSectionDescription> InCmd = readFilePattern();
  Opt.KeptSections.insert(Opt.KeptSections.end(), InCmd->Patterns.begin(),
                          InCmd->Patterns.end());
  Cmd->Commands.push_back(std::move(InCmd));
  expect(")");
}

void ScriptParser::readAlign(OutputSectionCommand *Cmd) {
  expect("(");
  Cmd->AlignExpr = readExpr();
  expect(")");
}

void ScriptParser::readOutputSectionDescription(StringRef OutSec) {
  OutputSectionCommand *Cmd = new OutputSectionCommand(OutSec);
  Opt.Commands.emplace_back(Cmd);

  // Read an address expression.
  // https://sourceware.org/binutils/docs/ld/Output-Section-Address.html#Output-Section-Address
  if (peek() != ":")
    Cmd->AddrExpr = readExpr();

  expect(":");

  if (skip("ALIGN"))
    readAlign(Cmd);

  // Parse constraints.
  if (skip("ONLY_IF_RO"))
    Cmd->Constraint = ConstraintKind::ReadOnly;
  if (skip("ONLY_IF_RW"))
    Cmd->Constraint = ConstraintKind::ReadWrite;
  expect("{");

  while (!Error && !skip("}")) {
    StringRef Tok = next();
    if (Tok == "*") {
      auto *InCmd = new InputSectionDescription();
      Cmd->Commands.emplace_back(InCmd);
      expect("(");
      while (!Error && !skip(")"))
        InCmd->Patterns.push_back(next());
    } else if (Tok == "KEEP") {
      readKeep(Cmd);
    } else if (Tok == "PROVIDE") {
      readProvide(false);
    } else if (Tok == "PROVIDE_HIDDEN") {
      readProvide(true);
    } else {
      setError("unknown command " + Tok);
    }
  }
  Cmd->Phdrs = readOutputSectionPhdrs();

  StringRef Tok = peek();
  if (Tok.startswith("=")) {
    if (!Tok.startswith("=0x")) {
      setError("filler should be a hexadecimal value");
      return;
    }
    Tok = Tok.substr(3);
    Cmd->Filler = parseHex(Tok);
    next();
  }
}

void ScriptParser::readProvide(bool Hidden) {
  expect("(");
  if (SymbolAssignment *Assignment = readAssignment(next())) {
    Assignment->Provide = true;
    Assignment->Hidden = Hidden;
  }
  expect(")");
  expect(";");
}

SymbolAssignment *ScriptParser::readAssignment(StringRef Name) {
  expect("=");
  Expr E = readExpr();
  auto *Cmd = new SymbolAssignment(Name, E);
  Opt.Commands.emplace_back(Cmd);
  return Cmd;
}

// This is an operator-precedence parser to parse a linker
// script expression.
Expr ScriptParser::readExpr() { return readExpr1(readPrimary(), 0); }

static uint64_t getSymbolValue(StringRef S) {
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
  if (S == "COMMONPAGESIZE" || S == "MAXPAGESIZE")
    return Target->PageSize;
  error("unknown constant: " + S);
  return 0;
}

Expr ScriptParser::readPrimary() {
  StringRef Tok = next();

  if (Tok == ".")
    return [](uint64_t Dot) { return Dot; };

  if (Tok == "(") {
    Expr E = readExpr();
    expect(")");
    return E;
  }

  // Built-in functions are parsed here.
  // https://sourceware.org/binutils/docs/ld/Builtin-Functions.html.
  if (Tok == "ALIGN") {
    expect("(");
    Expr E = readExpr();
    expect(")");
    return [=](uint64_t Dot) { return alignTo(Dot, E(Dot)); };
  }
  if (Tok == "CONSTANT") {
    expect("(");
    StringRef Tok = next();
    expect(")");
    return [=](uint64_t Dot) { return getConstant(Tok); };
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

  // Parse a symbol name or a number literal.
  uint64_t V = 0;
  if (Tok.getAsInteger(0, V)) {
    if (!isValidCIdentifier(Tok))
      setError("malformed number: " + Tok);
    return [=](uint64_t Dot) { return getSymbolValue(Tok); };
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

Expr ScriptParser::combine(StringRef Op, Expr L, Expr R) {
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
