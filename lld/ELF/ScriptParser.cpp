//===- ScriptParser.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the base parser class for linker script and dynamic
// list.
//
//===----------------------------------------------------------------------===//

#include "ScriptParser.h"
#include "Error.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;
using namespace lld;
using namespace lld::elf;

// Returns the line that the character S[Pos] is in.
static StringRef getLine(StringRef S, size_t Pos) {
  size_t Begin = S.rfind('\n', Pos);
  size_t End = S.find('\n', Pos);
  Begin = (Begin == StringRef::npos) ? 0 : Begin + 1;
  if (End == StringRef::npos)
    End = S.size();
  // rtrim for DOS-style newlines.
  return S.substr(Begin, End - Begin).rtrim();
}

void ScriptParserBase::printErrorPos() {
  StringRef Tok = Tokens[Pos == 0 ? 0 : Pos - 1];
  StringRef Line = getLine(Input, Tok.data() - Input.data());
  size_t Col = Tok.data() - Line.data();
  error(Line);
  error(std::string(Col, ' ') + "^");
}

// We don't want to record cascading errors. Keep only the first one.
void ScriptParserBase::setError(const Twine &Msg) {
  if (Error)
    return;
  if (Input.empty() || Tokens.empty()) {
    error(Msg);
  } else {
    error("line " + Twine(getPos()) + ": " + Msg);
    printErrorPos();
  }
  Error = true;
}

// Split S into linker script tokens.
std::vector<StringRef> ScriptParserBase::tokenize(StringRef S) {
  std::vector<StringRef> Ret;
  for (;;) {
    S = skipSpace(S);
    if (S.empty())
      return Ret;

    // Quoted token.
    if (S.startswith("\"")) {
      size_t E = S.find("\"", 1);
      if (E == StringRef::npos) {
        error("unclosed quote");
        return {};
      }
      Ret.push_back(S.substr(1, E - 1));
      S = S.substr(E + 1);
      continue;
    }

    // Unquoted token. This is more relaxed than tokens in C-like language,
    // so that you can write "file-name.cpp" as one bare token, for example.
    size_t Pos = S.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789_.$/\\~=+[]*?-:!<>^");

    // A character that cannot start a word (which is usually a
    // punctuation) forms a single character token.
    if (Pos == 0)
      Pos = 1;
    Ret.push_back(S.substr(0, Pos));
    S = S.substr(Pos);
  }
}

// Skip leading whitespace characters or comments.
StringRef ScriptParserBase::skipSpace(StringRef S) {
  for (;;) {
    if (S.startswith("/*")) {
      size_t E = S.find("*/", 2);
      if (E == StringRef::npos) {
        error("unclosed comment in a linker script");
        return "";
      }
      S = S.substr(E + 2);
      continue;
    }
    if (S.startswith("#")) {
      size_t E = S.find('\n', 1);
      if (E == StringRef::npos)
        E = S.size() - 1;
      S = S.substr(E + 1);
      continue;
    }
    size_t Size = S.size();
    S = S.ltrim();
    if (S.size() == Size)
      return S;
  }
}

// An erroneous token is handled as if it were the last token before EOF.
bool ScriptParserBase::atEOF() { return Error || Tokens.size() == Pos; }

StringRef ScriptParserBase::next() {
  if (Error)
    return "";
  if (atEOF()) {
    setError("unexpected EOF");
    return "";
  }
  return Tokens[Pos++];
}

StringRef ScriptParserBase::peek() {
  StringRef Tok = next();
  if (Error)
    return "";
  --Pos;
  return Tok;
}

bool ScriptParserBase::skip(StringRef Tok) {
  if (Error)
    return false;
  if (atEOF()) {
    setError("unexpected EOF");
    return false;
  }
  if (Tokens[Pos] != Tok)
    return false;
  ++Pos;
  return true;
}

void ScriptParserBase::expect(StringRef Expect) {
  if (Error)
    return;
  StringRef Tok = next();
  if (Tok != Expect)
    setError(Expect + " expected, but got " + Tok);
}

// Returns the current line number.
size_t ScriptParserBase::getPos() {
  if (Pos == 0)
    return 1;
  const char *Begin = Input.data();
  const char *Tok = Tokens[Pos - 1].data();
  return StringRef(Begin, Tok - Begin).count('\n') + 1;
}
