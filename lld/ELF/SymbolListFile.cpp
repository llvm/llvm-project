//===- SymbolListFile.cpp -------------------------------------------------===//
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

#include "SymbolListFile.h"
#include "Config.h"
#include "ScriptParser.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

// Parse the --dynamic-list argument.  A dynamic list is in the form
//
//  { symbol1; symbol2; [...]; symbolN };
//
// Multiple groups can be defined in the same file, and they are merged
// into a single group.

namespace {
class DynamicListParser final : public ScriptParserBase {
public:
  DynamicListParser(StringRef S) : ScriptParserBase(S) {}
  void run();
};
} // end anonymous namespace

void DynamicListParser::run() {
  while (!atEOF()) {
    expect("{");
    while (!Error) {
      Config->DynamicList.push_back(next());
      expect(";");
      if (skip("}"))
        break;
    }
    expect(";");
  }
}

void elf::parseDynamicList(MemoryBufferRef MB) {
  DynamicListParser(MB.getBuffer()).run();
}

// Parse the --version-script argument. We currently only accept the following
// version script syntax:
//
//  { [ global: symbol1; symbol2; [...]; symbolN; ] local: *; };
//
// No wildcards are supported, other than for the local entry. Symbol versioning
// is also not supported.

namespace {
class VersionScriptParser final : public ScriptParserBase {
public:
  VersionScriptParser(StringRef S) : ScriptParserBase(S) {}

  void run();

private:
  void parseExtern(std::vector<SymbolVersion> *Globals);
  void parseVersion(StringRef VerStr);
  void parseGlobal(StringRef VerStr);
  void parseLocal();
};
} // end anonymous namespace

void VersionScriptParser::parseVersion(StringRef VerStr) {
  // Identifiers start at 2 because 0 and 1 are reserved
  // for VER_NDX_LOCAL and VER_NDX_GLOBAL constants.
  size_t VersionId = Config->VersionDefinitions.size() + 2;
  Config->VersionDefinitions.push_back({VerStr, VersionId});

  if (skip("global:") || peek() != "local:")
    parseGlobal(VerStr);
  if (skip("local:"))
    parseLocal();
  expect("}");

  // Each version may have a parent version. For example, "Ver2" defined as
  // "Ver2 { global: foo; local: *; } Ver1;" has "Ver1" as a parent. This
  // version hierarchy is, probably against your instinct, purely for human; the
  // runtime doesn't care about them at all. In LLD, we simply skip the token.
  if (!VerStr.empty() && peek() != ";")
    next();
  expect(";");
}

void VersionScriptParser::parseLocal() {
  Config->DefaultSymbolVersion = VER_NDX_LOCAL;
  expect("*");
  expect(";");
}

void VersionScriptParser::parseExtern(std::vector<SymbolVersion> *Globals) {
  expect("C++");
  expect("{");

  for (;;) {
    if (peek() == "}" || Error)
      break;
    Globals->push_back({next(), true});
    expect(";");
  }

  expect("}");
  expect(";");
}

void VersionScriptParser::parseGlobal(StringRef VerStr) {
  std::vector<SymbolVersion> *Globals;
  if (VerStr.empty())
    Globals = &Config->VersionScriptGlobals;
  else
    Globals = &Config->VersionDefinitions.back().Globals;

  for (;;) {
    if (skip("extern"))
      parseExtern(Globals);

    StringRef Cur = peek();
    if (Cur == "}" || Cur == "local:" || Error)
      return;
    next();
    Globals->push_back({Cur, false});
    expect(";");
  }
}

void VersionScriptParser::run() {
  StringRef Msg = "anonymous version definition is used in "
                  "combination with other version definitions";
  if (skip("{")) {
    parseVersion("");
    if (!atEOF())
      setError(Msg);
    return;
  }

  while (!atEOF() && !Error) {
    StringRef VerStr = next();
    if (VerStr == "{") {
      setError(Msg);
      return;
    }
    expect("{");
    parseVersion(VerStr);
  }
}

void elf::parseVersionScript(MemoryBufferRef MB) {
  VersionScriptParser(MB.getBuffer()).run();
}
