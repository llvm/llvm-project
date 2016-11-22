//===- ScriptParser.h -------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SCRIPT_PARSER_H
#define LLD_ELF_SCRIPT_PARSER_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <utility>
#include <vector>

namespace lld {
namespace elf {

class ScriptParserBase {
public:
  explicit ScriptParserBase(MemoryBufferRef MB);

  void setError(const Twine &Msg);
  void tokenize(MemoryBufferRef MB);
  static StringRef skipSpace(StringRef S);
  bool atEOF();
  StringRef next();
  StringRef peek();
  void skip();
  bool consume(StringRef Tok);
  void expect(StringRef Expect);

  std::vector<MemoryBufferRef> MBs;
  std::vector<StringRef> Tokens;
  size_t Pos = 0;
  bool Error = false;

private:
  MemoryBufferRef currentBuffer();
};

} // namespace elf
} // namespace lld

#endif
