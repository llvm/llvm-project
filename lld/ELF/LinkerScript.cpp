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

#include "Config.h"
#include "Driver.h"
#include "SymbolTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace lld;
using namespace lld::elf2;

namespace {
class LinkerScript {
public:
  LinkerScript(BumpPtrAllocator *A, StringRef S, bool B)
      : Saver(*A), Tokens(tokenize(S)), IsUnderSysroot(B) {}
  void run();

private:
  static std::vector<StringRef> tokenize(StringRef S);
  static StringRef skipSpace(StringRef S);
  StringRef next();
  bool skip(StringRef Tok);
  bool atEOF() { return Tokens.size() == Pos; }
  void expect(StringRef Expect);

  void addFile(StringRef Path);

  void readAsNeeded();
  void readEntry();
  void readExtern();
  void readGroup();
  void readInclude();
  void readOutput();
  void readOutputArch();
  void readOutputFormat();
  void readSearchDir();
  void readSections();

  void readOutputSectionDescription();

  StringSaver Saver;
  std::vector<StringRef> Tokens;
  size_t Pos = 0;
  bool IsUnderSysroot;
};
}

void LinkerScript::run() {
  while (!atEOF()) {
    StringRef Tok = next();
    if (Tok == ";")
      continue;
    if (Tok == "ENTRY") {
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
    } else if (Tok == "SEARCH_DIR") {
      readSearchDir();
    } else if (Tok == "SECTIONS") {
      readSections();
    } else {
      error("unknown directive: " + Tok);
    }
  }
}

// Split S into linker script tokens.
std::vector<StringRef> LinkerScript::tokenize(StringRef S) {
  std::vector<StringRef> Ret;
  for (;;) {
    S = skipSpace(S);
    if (S.empty())
      return Ret;

    // Quoted token
    if (S.startswith("\"")) {
      size_t E = S.find("\"", 1);
      if (E == StringRef::npos)
        error("unclosed quote");
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
StringRef LinkerScript::skipSpace(StringRef S) {
  for (;;) {
    if (S.startswith("/*")) {
      size_t E = S.find("*/", 2);
      if (E == StringRef::npos)
        error("unclosed comment in a linker script");
      S = S.substr(E + 2);
      continue;
    }
    size_t Size = S.size();
    S = S.ltrim();
    if (S.size() == Size)
      return S;
  }
}

StringRef LinkerScript::next() {
  if (atEOF())
    error("unexpected EOF");
  return Tokens[Pos++];
}

bool LinkerScript::skip(StringRef Tok) {
  if (atEOF())
    error("unexpected EOF");
  if (Tok != Tokens[Pos])
    return false;
  ++Pos;
  return true;
}

void LinkerScript::expect(StringRef Expect) {
  StringRef Tok = next();
  if (Tok != Expect)
    error(Expect + " expected, but got " + Tok);
}

void LinkerScript::addFile(StringRef S) {
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
    Driver->addFile(searchLibrary(S.substr(2)));
  } else if (sys::fs::exists(S)) {
    Driver->addFile(S);
  } else {
    std::string Path = findFromSearchPaths(S);
    if (Path.empty())
      error("Unable to find " + S);
    Driver->addFile(Saver.save(Path));
  }
}

void LinkerScript::readAsNeeded() {
  expect("(");
  bool Orig = Config->AsNeeded;
  Config->AsNeeded = true;
  for (;;) {
    StringRef Tok = next();
    if (Tok == ")")
      break;
    addFile(Tok);
  }
  Config->AsNeeded = Orig;
}

void LinkerScript::readEntry() {
  // -e <symbol> takes predecence over ENTRY(<symbol>).
  expect("(");
  StringRef Tok = next();
  if (Config->Entry.empty())
    Config->Entry = Tok;
  expect(")");
}

void LinkerScript::readExtern() {
  expect("(");
  for (;;) {
    StringRef Tok = next();
    if (Tok == ")")
      return;
    Config->Undefined.push_back(Tok);
  }
}

void LinkerScript::readGroup() {
  expect("(");
  for (;;) {
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

void LinkerScript::readInclude() {
  StringRef Tok = next();
  auto MBOrErr = MemoryBuffer::getFile(Tok);
  error(MBOrErr, "cannot open " + Tok);
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  StringRef S = Saver.save(MB->getMemBufferRef().getBuffer());
  std::vector<StringRef> V = tokenize(S);
  Tokens.insert(Tokens.begin() + Pos, V.begin(), V.end());
}

void LinkerScript::readOutput() {
  // -o <file> takes predecence over OUTPUT(<file>).
  expect("(");
  StringRef Tok = next();
  if (Config->OutputFile.empty())
    Config->OutputFile = Tok;
  expect(")");
}

void LinkerScript::readOutputArch() {
  // Error checking only for now.
  expect("(");
  next();
  expect(")");
}

void LinkerScript::readOutputFormat() {
  // Error checking only for now.
  expect("(");
  next();
  StringRef Tok = next();
  if (Tok == ")")
   return;
  if (Tok != ",")
    error("unexpected token: " + Tok);
  next();
  expect(",");
  next();
  expect(")");
}

void LinkerScript::readSearchDir() {
  expect("(");
  Config->SearchPaths.push_back(next());
  expect(")");
}

void LinkerScript::readSections() {
  expect("{");
  while (!skip("}"))
    readOutputSectionDescription();
}

void LinkerScript::readOutputSectionDescription() {
  StringRef Name = next();
  std::vector<StringRef> &InputSections = Config->OutputSections[Name];

  expect(":");
  expect("{");
  while (!skip("}")) {
    next(); // Skip input file name.
    expect("(");
    while (!skip(")"))
      InputSections.push_back(next());
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
void elf2::readLinkerScript(BumpPtrAllocator *A, MemoryBufferRef MB) {
  StringRef Path = MB.getBufferIdentifier();
  LinkerScript(A, MB.getBuffer(), isUnderSysroot(Path)).run();
}
