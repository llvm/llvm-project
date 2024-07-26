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
    // The remaining content to parse and the filename.
    StringRef s, filename;
    const char *begin = nullptr;
    Buffer() = default;
    Buffer(MemoryBufferRef mb)
        : s(mb.getBuffer()), filename(mb.getBufferIdentifier()),
          begin(mb.getBufferStart()) {}
  };
  // The current buffer and parent buffers due to INCLUDE.
  Buffer curBuf;
  SmallVector<Buffer, 0> buffers;

  // The token before the last next().
  StringRef prevTok;
  // Rules for what is a token are different when we are in an expression.
  // curTok holds the cached return value of peek() and is invalid when the
  // expression state changes.
  StringRef curTok;
  // The inExpr state when curTok is cached.
  bool curTokState = false;
  bool eof = false;

public:
  explicit ScriptLexer(MemoryBufferRef mb);

  void setError(const Twine &msg);
  void lex();
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
  StringRef getLine();
  size_t getLineNumber();
  size_t getColumnNumber();
};

} // namespace lld::elf

#endif
