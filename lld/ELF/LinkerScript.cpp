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
// It does not construct an AST but consume linker script directives directly.
// Results are written to Driver or Config object.
//
//===----------------------------------------------------------------------===//

#include "LinkerScript.h"
#include "Config.h"
#include "Driver.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "ScriptParser.h"
#include "SymbolTable.h"
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

LinkerScript *elf::Script;

static uint64_t getInteger(StringRef S) {
  uint64_t V;
  if (S.getAsInteger(0, V)) {
    error("malformed number: " + S);
    return 0;
  }
  return V;
}

// Evaluates the expression given by list of tokens.
uint64_t LinkerScript::evaluate(std::vector<StringRef> &Tokens,
                                uint64_t LocCounter) {
  uint64_t Result = 0;
  for (size_t I = 0, E = Tokens.size(); I < E; ++I) {
    // Each second token should be '+' as this is the
    // only operator we support now.
    if (I % 2 == 1) {
      if (Tokens[I] == "+")
        continue;
      error("error in location counter expression");
      return 0;
    }

    StringRef Tok = Tokens[I];
    if (Tok == ".")
      Result += LocCounter;
    else
      Result += getInteger(Tok);
  }
  return Result;
}

template <class ELFT>
SectionRule *LinkerScript::find(InputSectionBase<ELFT> *S) {
  for (SectionRule &R : Sections)
    if (R.match(S))
      return &R;
  return nullptr;
}

template <class ELFT>
StringRef LinkerScript::getOutputSection(InputSectionBase<ELFT> *S) {
  SectionRule *R = find(S);
  return R ? R->Dest : "";
}

template <class ELFT>
bool LinkerScript::isDiscarded(InputSectionBase<ELFT> *S) {
  return getOutputSection(S) == "/DISCARD/";
}

template <class ELFT> bool LinkerScript::shouldKeep(InputSectionBase<ELFT> *S) {
  SectionRule *R = find(S);
  return R && R->Keep;
}

template <class ELFT>
static OutputSectionBase<ELFT> *
findSection(std::vector<OutputSectionBase<ELFT> *> &V, StringRef Name) {
  for (OutputSectionBase<ELFT> *Sec : V)
    if (Sec->getName() == Name)
      return Sec;
  return nullptr;
}

template <class ELFT>
void LinkerScript::assignAddresses(
    std::vector<OutputSectionBase<ELFT> *> &Sections) {
  typedef typename ELFT::uint uintX_t;

  // Orphan sections are sections present in the input files which
  // are not explicitly placed into the output file by the linker script.
  // We place orphan sections at end of file.
  // Other linkers places them using some heuristics as described in
  // https://sourceware.org/binutils/docs/ld/Orphan-Sections.html#Orphan-Sections.
  for (OutputSectionBase<ELFT> *Sec : Sections) {
    StringRef Name = Sec->getName();
    auto I = std::find(SectionOrder.begin(), SectionOrder.end(), Name);
    if (I == SectionOrder.end())
      Commands.push_back({SectionKind, {}, Name});
  }

  // Assign addresses as instructed by linker script SECTIONS sub-commands.
  uintX_t ThreadBssOffset = 0;
  uintX_t VA =
      Out<ELFT>::ElfHeader->getSize() + Out<ELFT>::ProgramHeaders->getSize();

  for (SectionsCommand &Cmd : Commands) {
    if (Cmd.Kind == ExprKind) {
      VA = evaluate(Cmd.Expr, VA);
      continue;
    }

    OutputSectionBase<ELFT> *Sec = findSection(Sections, Cmd.SectionName);
    if (!Sec)
      continue;

    if ((Sec->getFlags() & SHF_TLS) && Sec->getType() == SHT_NOBITS) {
      uintX_t TVA = VA + ThreadBssOffset;
      TVA = alignTo(TVA, Sec->getAlign());
      Sec->setVA(TVA);
      ThreadBssOffset = TVA - VA + Sec->getSize();
      continue;
    }

    if (Sec->getFlags() & SHF_ALLOC) {
      VA = alignTo(VA, Sec->getAlign());
      Sec->setVA(VA);
      VA += Sec->getSize();
      continue;
    }
  }
}

ArrayRef<uint8_t> LinkerScript::getFiller(StringRef Name) {
  auto I = Filler.find(Name);
  if (I == Filler.end())
    return {};
  return I->second;
}

// A compartor to sort output sections. Returns -1 or 1 if both
// A and B are mentioned in linker scripts. Otherwise, returns 0
// to use the default rule which is implemented in Writer.cpp.
int LinkerScript::compareSections(StringRef A, StringRef B) {
  auto E = SectionOrder.end();
  auto I = std::find(SectionOrder.begin(), E, A);
  auto J = std::find(SectionOrder.begin(), E, B);
  if (I == E || J == E)
    return 0;
  return I < J ? -1 : 1;
}

// Returns true if S matches T. S can contain glob meta-characters.
// The asterisk ('*') matches zero or more characacters, and the question
// mark ('?') matches one character.
static bool matchStr(StringRef S, StringRef T) {
  for (;;) {
    if (S.empty())
      return T.empty();
    if (S[0] == '*') {
      S = S.substr(1);
      if (S.empty())
        // Fast path. If a pattern is '*', it matches anything.
        return true;
      for (size_t I = 0, E = T.size(); I < E; ++I)
        if (matchStr(S, T.substr(I)))
          return true;
      return false;
    }
    if (T.empty() || (S[0] != T[0] && S[0] != '?'))
      return false;
    S = S.substr(1);
    T = T.substr(1);
  }
}

template <class ELFT> bool SectionRule::match(InputSectionBase<ELFT> *S) {
  return matchStr(SectionPattern, S->getSectionName());
}

class elf::ScriptParser final : public elf::ScriptParserBase {
  typedef void (ScriptParser::*Handler)();

public:
  ScriptParser(BumpPtrAllocator *A, StringRef S, bool B)
      : ScriptParserBase(S), Saver(*A), IsUnderSysroot(B) {}

  void run() override;

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
  void readSearchDir();
  void readSections();

  void readLocationCounterValue();
  void readOutputSectionDescription();
  void readSectionPatterns(StringRef OutSec, bool Keep);

  StringSaver Saver;
  const static StringMap<Handler> Cmd;
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

void ScriptParser::readSearchDir() {
  expect("(");
  Config->SearchPaths.push_back(next());
  expect(")");
}

void ScriptParser::readSections() {
  Script->DoLayout = true;
  expect("{");
  while (!Error && !skip("}")) {
    StringRef Tok = peek();
    if (Tok == ".")
      readLocationCounterValue();
    else
      readOutputSectionDescription();
  }
}

void ScriptParser::readSectionPatterns(StringRef OutSec, bool Keep) {
  expect("(");
  while (!Error && !skip(")"))
    Script->Sections.emplace_back(OutSec, next(), Keep);
}

void ScriptParser::readLocationCounterValue() {
  expect(".");
  expect("=");
  Script->Commands.push_back({ExprKind, {}, ""});
  SectionsCommand &Cmd = Script->Commands.back();
  while (!Error) {
    StringRef Tok = next();
    if (Tok == ";")
      break;
    Cmd.Expr.push_back(Tok);
  }
  if (Cmd.Expr.empty())
    error("error in location counter expression");
}

void ScriptParser::readOutputSectionDescription() {
  StringRef OutSec = next();
  Script->SectionOrder.push_back(OutSec);
  Script->Commands.push_back({SectionKind, {}, OutSec});
  expect(":");
  expect("{");
  while (!Error && !skip("}")) {
    StringRef Tok = next();
    if (Tok == "*") {
      readSectionPatterns(OutSec, false);
    } else if (Tok == "KEEP") {
      expect("(");
      next(); // Skip *
      readSectionPatterns(OutSec, true);
      expect(")");
    } else {
      setError("unknown command " + Tok);
    }
  }
  StringRef Tok = peek();
  if (Tok.startswith("=")) {
    if (!Tok.startswith("=0x")) {
      setError("filler should be a hexadecimal value");
      return;
    }
    Tok = Tok.substr(3);
    Script->Filler[OutSec] = parseHex(Tok);
    next();
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

// Entry point. The other functions or classes are private to this file.
void LinkerScript::read(MemoryBufferRef MB) {
  StringRef Path = MB.getBufferIdentifier();
  ScriptParser(&Alloc, MB.getBuffer(), isUnderSysroot(Path)).run();
}

template StringRef LinkerScript::getOutputSection(InputSectionBase<ELF32LE> *);
template StringRef LinkerScript::getOutputSection(InputSectionBase<ELF32BE> *);
template StringRef LinkerScript::getOutputSection(InputSectionBase<ELF64LE> *);
template StringRef LinkerScript::getOutputSection(InputSectionBase<ELF64BE> *);

template bool LinkerScript::isDiscarded(InputSectionBase<ELF32LE> *);
template bool LinkerScript::isDiscarded(InputSectionBase<ELF32BE> *);
template bool LinkerScript::isDiscarded(InputSectionBase<ELF64LE> *);
template bool LinkerScript::isDiscarded(InputSectionBase<ELF64BE> *);

template bool LinkerScript::shouldKeep(InputSectionBase<ELF32LE> *);
template bool LinkerScript::shouldKeep(InputSectionBase<ELF32BE> *);
template bool LinkerScript::shouldKeep(InputSectionBase<ELF64LE> *);
template bool LinkerScript::shouldKeep(InputSectionBase<ELF64BE> *);

template void
LinkerScript::assignAddresses(std::vector<OutputSectionBase<ELF32LE> *> &);
template void
LinkerScript::assignAddresses(std::vector<OutputSectionBase<ELF32BE> *> &);
template void
LinkerScript::assignAddresses(std::vector<OutputSectionBase<ELF64LE> *> &);
template void
LinkerScript::assignAddresses(std::vector<OutputSectionBase<ELF64BE> *> &);
