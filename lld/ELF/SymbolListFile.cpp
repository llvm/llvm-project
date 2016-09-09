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
#include "Strings.h"
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
      Config->DynamicList.push_back(unquote(next()));
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
