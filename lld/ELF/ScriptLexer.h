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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <vector>

namespace lld::elf {

class ScriptLexer {
protected:
  struct Buffer {
    // The unparsed buffer and the filename.
    StringRef s, filename;
  };
  // The current buffer and parent buffers due to INCLUDE.
  Buffer cur;
  SmallVector<Buffer, 0> buffers;

  // The token before the last next().
  StringRef prevTok;
  // The cache value of peek(). This is valid if curTokState and inExpr match.
  StringRef curTok;
  // The inExpr state when curTok is cached.
  bool curTokState = false;
  bool eof = false;

public:
  explicit ScriptLexer(MemoryBufferRef mb);

  void setError(const Twine &msg);
  void lexToken();
  StringRef skipSpace(StringRef s);
  bool atEOF();
  StringRef next();
  StringRef peek();
  void skip();
  bool consume(StringRef tok);
  void expect(StringRef expect);
  std::string getCurrentLocation();
  MemoryBufferRef getCurrentMB();

  std::vector<MemoryBufferRef> mbs;
  bool inExpr = false;

  size_t lastLineNumber = 0;
  size_t lastLineNumberOffset = 0;

private:
  void maybeSplitExpr();
  StringRef getLine();
  size_t getLineNumber();
  size_t getColumnNumber();
};

} // namespace lld::elf

#endif
