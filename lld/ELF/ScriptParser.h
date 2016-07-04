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
#include <utility>
#include <vector>

namespace lld {
namespace elf {

class ScriptParserBase {
public:
  explicit ScriptParserBase(StringRef S) : Input(S), Tokens(tokenize(S)) {}
  explicit ScriptParserBase(std::vector<StringRef> Tokens)
      : Input(""), Tokens(std::move(Tokens)) {}

protected:
  void setError(const Twine &Msg);
  static std::vector<StringRef> tokenize(StringRef S);
  static StringRef skipSpace(StringRef S);
  bool atEOF();
  StringRef next();
  StringRef peek();
  bool skip(StringRef Tok);
  void expect(StringRef Expect);

  size_t getPos();
  void printErrorPos();

  StringRef Input;
  std::vector<StringRef> Tokens;
  size_t Pos = 0;
  bool Error = false;
};

} // namespace elf
} // namespace lld

#endif
