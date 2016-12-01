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

// Returns a whole line containing the current token.
StringRef ScriptParserBase::getLine() {
  StringRef S = getCurrentMB().getBuffer();
  StringRef Tok = Tokens[Pos - 1];

  size_t Pos = S.rfind('\n', Tok.data() - S.data());
  if (Pos != StringRef::npos)
    S = S.substr(Pos + 1);
  return S.substr(0, S.find_first_of("\r\n"));
}

// Returns 1-based line number of the current token.
size_t ScriptParserBase::getLineNumber() {
  StringRef S = getCurrentMB().getBuffer();
  StringRef Tok = Tokens[Pos - 1];
  return S.substr(0, Tok.data() - S.data()).count('\n') + 1;
}

// Returns 0-based column number of the current token.
size_t ScriptParserBase::getColumnNumber() {
  StringRef Tok = Tokens[Pos - 1];
  return Tok.data() - getLine().data();
}

std::string ScriptParserBase::getCurrentLocation() {
  std::string Filename = getCurrentMB().getBufferIdentifier();
  if (!Pos)
    return Filename;
  return (Filename + ":" + Twine(getLineNumber())).str();
}

ScriptParserBase::ScriptParserBase(MemoryBufferRef MB) { tokenize(MB); }

// We don't want to record cascading errors. Keep only the first one.
void ScriptParserBase::setError(const Twine &Msg) {
  if (Error)
    return;
  Error = true;

  if (!Pos) {
    error(getCurrentLocation() + ": " + Msg);
    return;
  }

  std::string S = getCurrentLocation() + ": ";
  error(S + Msg);
  error(S + getLine());
  error(S + std::string(getColumnNumber(), ' ') + "^");
}

// Split S into linker script tokens.
void ScriptParserBase::tokenize(MemoryBufferRef MB) {
  std::vector<StringRef> Vec;
  MBs.push_back(MB);
  StringRef S = MB.getBuffer();
  StringRef Begin = S;

  for (;;) {
    S = skipSpace(S);
    if (S.empty())
      break;

    // Quoted token. Note that double-quote characters are parts of a token
    // because, in a glob match context, only unquoted tokens are interpreted
    // as glob patterns. Double-quoted tokens are literal patterns in that
    // context.
    if (S.startswith("\"")) {
      size_t E = S.find("\"", 1);
      if (E == StringRef::npos) {
        StringRef Filename = MB.getBufferIdentifier();
        size_t Lineno = Begin.substr(0, S.data() - Begin.data()).count('\n');
        error(Filename + ":" + Twine(Lineno + 1) + ": unclosed quote");
        return;
      }

      Vec.push_back(S.take_front(E + 1));
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
    Vec.push_back(S.substr(0, Pos));
    S = S.substr(Pos);
  }

  Tokens.insert(Tokens.begin() + Pos, Vec.begin(), Vec.end());
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

bool ScriptParserBase::consume(StringRef Tok) {
  if (peek() == Tok) {
    skip();
    return true;
  }
  return false;
}

void ScriptParserBase::skip() { (void)next(); }

void ScriptParserBase::expect(StringRef Expect) {
  if (Error)
    return;
  StringRef Tok = next();
  if (Tok != Expect)
    setError(Expect + " expected, but got " + Tok);
}

// Returns true if S encloses T.
static bool encloses(StringRef S, StringRef T) {
  return S.bytes_begin() <= T.bytes_begin() && T.bytes_end() <= S.bytes_end();
}

MemoryBufferRef ScriptParserBase::getCurrentMB() {
  // Find input buffer containing the current token.
  assert(!MBs.empty());
  if (!Pos)
    return MBs[0];

  for (MemoryBufferRef MB : MBs)
    if (encloses(MB.getBuffer(), Tokens[Pos - 1]))
      return MB;
  llvm_unreachable("getCurrentMB: failed to find a token");
}
