//===- ScriptLexer.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SCRIPT_LEXER_H
#define LLD_ELF_SCRIPT_LEXER_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <utility>
#include <vector>

namespace lld {
namespace elf {

class ScriptLexer {
public:
  explicit ScriptLexer(MemoryBufferRef MB);

  void setError(const Twine &Msg);
  void tokenize(MemoryBufferRef MB);
  static StringRef skipSpace(StringRef S);
  bool atEOF();
  StringRef next();
  StringRef peek();
  StringRef peek2();
  void skip();
  bool consume(StringRef Tok);
  void expect(StringRef Expect);
  bool consumeLabel(StringRef Tok);
  std::string getCurrentLocation();

  std::vector<MemoryBufferRef> MBs;
  std::vector<StringRef> Tokens;
  bool InExpr = false;
  size_t Pos = 0;

private:
  void maybeSplitExpr();
  StringRef getLine();
  size_t getLineNumber();
  size_t getColumnNumber();

  MemoryBufferRef getCurrentMB();
};

} // namespace elf
} // namespace lld

#endif
