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
using namespace lld;
using namespace lld::elf;

// Parse the --dynamic-list argument.  A dynamic list is in the form
//
//  { symbol1; symbol2; [...]; symbolN };
//
// Multiple groups can be defined in the same file and they are merged
// in only one definition.

class DynamicListParser final : public ScriptParserBase {
public:
  DynamicListParser(StringRef S) : ScriptParserBase(S) {}

  void run();

private:
  void readGroup();
};

// Parse the default group definition using C language symbol name.
void DynamicListParser::readGroup() {
  expect("{");
  while (!Error) {
    Config->DynamicList.push_back(next());
    expect(";");
    if (peek() == "}") {
      next();
      break;
    }
  }
  expect(";");
}

void DynamicListParser::run() {
  while (!atEOF())
    readGroup();
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

class VersionScriptParser final : public ScriptParserBase {
public:
  VersionScriptParser(StringRef S) : ScriptParserBase(S) {}

  void run();

private:
  void parseVersion();
};

void VersionScriptParser::parseVersion() {
  expect("{");
  if (peek() == "global:") {
    next();
    while (!Error) {
      Config->VersionScriptGlobals.push_back(next());
      expect(";");
      if (peek() == "local:")
        break;
    }
  }
  expect("local:");
  expect("*");
  expect(";");
  expect("}");
  expect(";");
}

void VersionScriptParser::run() {
  StringRef Msg = "anonymous version definition is used in "
                  "combination with other version definitions";
  if (peek() == "{") {
    parseVersion();
    if (!atEOF())
      setError(Msg);
    return;
  }

  while (!atEOF() && !Error) {
    if (next() == "{") {
      setError(Msg);
      return;
    }
    parseVersion();
  }
}

void elf::parseVersionScript(MemoryBufferRef MB) {
  VersionScriptParser(MB.getBuffer()).run();
}
