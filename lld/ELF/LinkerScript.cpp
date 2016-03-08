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
#include "SymbolTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::object;
using namespace lld;
using namespace lld::elf;

LinkerScript *elf::Script;

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

class elf::ScriptParser {
  typedef void (ScriptParser::*Handler)();

public:
  ScriptParser(BumpPtrAllocator *A, StringRef S, bool B)
      : Saver(*A), Tokens(tokenize(S)), IsUnderSysroot(B) {}

  void run();

private:
  void setError(const Twine &Msg);
  static std::vector<StringRef> tokenize(StringRef S);
  static StringRef skipSpace(StringRef S);
  bool atEOF();
  StringRef next();
  StringRef peek();
  bool skip(StringRef Tok);
  void expect(StringRef Expect);

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

  void readOutputSectionDescription();
  void readSectionPatterns(StringRef OutSec, bool Keep);

  std::vector<uint8_t> parseHex(StringRef S);

  StringSaver Saver;
  std::vector<StringRef> Tokens;
  const static StringMap<Handler> Cmd;
  size_t Pos = 0;
  bool IsUnderSysroot;
  bool Error = false;
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

// We don't want to record cascading errors. Keep only the first one.
void ScriptParser::setError(const Twine &Msg) {
  if (Error)
    return;
  error(Msg);
  Error = true;
}

// Split S into linker script tokens.
std::vector<StringRef> ScriptParser::tokenize(StringRef S) {
  std::vector<StringRef> Ret;
  for (;;) {
    S = skipSpace(S);
    if (S.empty())
      return Ret;

    // Quoted token
    if (S.startswith("\"")) {
      size_t E = S.find("\"", 1);
      if (E == StringRef::npos) {
        error("unclosed quote");
        return {};
      }
      Ret.push_back(S.substr(1, E - 1));
      S = S.substr(E + 1);
      continue;
    }

    // Unquoted token
    size_t Pos = S.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789_.$/\\~=+[]*?-:");
    // A character that cannot start a word (which is usually a
    // punctuation) forms a single character token.
    if (Pos == 0)
      Pos = 1;
    Ret.push_back(S.substr(0, Pos));
    S = S.substr(Pos);
  }
}

// Skip leading whitespace characters or /**/-style comments.
StringRef ScriptParser::skipSpace(StringRef S) {
  for (;;) {
    if (S.startswith("/*")) {
      size_t E = S.find("*/", 2);
      if (E == StringRef::npos) {
        error("unclosed comment in a linker script");
        return "";
      }
      S = S.substr(E + 2);
      continue;
    }
    size_t Size = S.size();
    S = S.ltrim();
    if (S.size() == Size)
      return S;
  }
}

// An errneous token is handled as if it were the last token before EOF.
bool ScriptParser::atEOF() { return Error || Tokens.size() == Pos; }

StringRef ScriptParser::next() {
  if (Error)
    return "";
  if (atEOF()) {
    setError("unexpected EOF");
    return "";
  }
  return Tokens[Pos++];
}

StringRef ScriptParser::peek() {
  StringRef Tok = next();
  if (Error)
    return "";
  --Pos;
  return Tok;
}

bool ScriptParser::skip(StringRef Tok) {
  if (Error)
    return false;
  if (atEOF()) {
    setError("unexpected EOF");
    return false;
  }
  if (Tok != Tokens[Pos])
    return false;
  ++Pos;
  return true;
}

void ScriptParser::expect(StringRef Expect) {
  if (Error)
    return;
  StringRef Tok = next();
  if (Tok != Expect)
    setError(Expect + " expected, but got " + Tok);
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
      setError("Unable to find " + S);
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
  StringRef Path = next();
  if (!Config->Nostdlib)
    Config->SearchPaths.push_back(Path);
  expect(")");
}

void ScriptParser::readSections() {
  expect("{");
  while (!Error && !skip("}"))
    readOutputSectionDescription();
}

void ScriptParser::readSectionPatterns(StringRef OutSec, bool Keep) {
  expect("(");
  while (!Error && !skip(")"))
    Script->Sections.emplace_back(OutSec, next(), Keep);
}

std::vector<uint8_t> ScriptParser::parseHex(StringRef S) {
  std::vector<uint8_t> Hex;
  while (!S.empty()) {
    StringRef B = S.substr(0, 2);
    S = S.substr(2);
    uint8_t H;
    if (B.getAsInteger(16, H)) {
      setError("Not a HEX value: " + B);
      return {};
    }
    Hex.push_back(H);
  }
  return Hex;
}

void ScriptParser::readOutputSectionDescription() {
  StringRef OutSec = next();
  Script->SectionOrder.push_back(OutSec);
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
      setError("Unknown command " + Tok);
    }
  }
  StringRef Tok = peek();
  if (Tok.startswith("=")) {
    if (!Tok.startswith("=0x")) {
      setError("Filler should be a HEX value");
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
