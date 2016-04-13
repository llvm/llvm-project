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

#include "DynamicList.h"
#include "Config.h"
#include "ScriptParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

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

  void run() override;

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
