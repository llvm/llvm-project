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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <vector>

namespace lld::elf {
struct Ctx;

class ScriptLexer {
protected:
  struct Buffer {
    // The remaining content to parse and the filename.
    StringRef s, filename;
    const char *begin = nullptr;
    size_t lineNumber = 1;
    // True if the script is opened as an absolute path under the --sysroot
    // directory.
    bool isUnderSysroot = false;

    Buffer() = default;
    Buffer(Ctx &ctx, MemoryBufferRef mb);
  };
  Ctx &ctx;
  // The current buffer and parent buffers due to INCLUDE.
  Buffer curBuf;
  SmallVector<Buffer, 0> buffers;

  // Used to detect INCLUDE() cycles.
  llvm::DenseSet<StringRef> activeFilenames;

  enum class State {
    Script,
    Expr,
  };

  struct Token {
    StringRef str;
    explicit operator bool() const { return !str.empty(); }
    operator StringRef() const { return str; }
  };

  // The token before the last next().
  StringRef prevTok;
  // Rules for what is a token are different when we are in an expression.
  // curTok holds the cached return value of peek() and is invalid when the
  // expression state changes.
  StringRef curTok;
  size_t prevTokLine = 1;
  // The lex state when curTok is cached.
  State curTokState = State::Script;
  State lexState = State::Script;
  bool eof = false;

public:
  explicit ScriptLexer(Ctx &ctx, MemoryBufferRef mb);

  void setError(const Twine &msg);
  void lex();
  StringRef skipSpace(StringRef s);
  bool atEOF();
  StringRef next();
  StringRef peek();
  void skip();
  bool consume(StringRef tok);
  void expect(StringRef expect);
  Token till(StringRef tok);
  std::string getCurrentLocation();
  MemoryBufferRef getCurrentMB();

  std::vector<MemoryBufferRef> mbs;

private:
  StringRef getLine();
  size_t getColumnNumber();
};

} // namespace lld::elf

#endif
