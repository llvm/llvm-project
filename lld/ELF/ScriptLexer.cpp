//===- ScriptLexer.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a lexer for the linker script.
//
// The linker script's grammar is not complex but ambiguous due to the
// lack of the formal specification of the language. What we are trying to
// do in this and other files in LLD is to make a "reasonable" linker
// script processor.
//
// Among simplicity, compatibility and efficiency, we put the most
// emphasis on simplicity when we wrote this lexer. Compatibility with the
// GNU linkers is important, but we did not try to clone every tiny corner
// case of their lexers, as even ld.bfd and ld.gold are subtly different
// in various corner cases. We do not care much about efficiency because
// the time spent in parsing linker scripts is usually negligible.
//
// Our grammar of the linker script is LL(1).
//
// Overall, this lexer works fine for most linker scripts. There might
// be room for improving compatibility, but that's probably not at the
// top of our todo list.
//
//===----------------------------------------------------------------------===//

#include "ScriptLexer.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

using namespace llvm;
using namespace lld;
using namespace lld::elf;

// Returns a whole line containing the current token.
StringRef ScriptLexer::getLine() {
  StringRef s = getCurrentMB().getBuffer();

  size_t pos = s.rfind('\n', prevTok.data() - s.data());
  if (pos != StringRef::npos)
    s = s.substr(pos + 1);
  return s.substr(0, s.find_first_of("\r\n"));
}

// Returns 1-based line number of the current token.
size_t ScriptLexer::getLineNumber() {
  if (prevTok.empty())
    return 1;
  StringRef s = getCurrentMB().getBuffer();
  const size_t tokOffset = prevTok.data() - s.data();

  // For the first token, or when going backwards, start from the beginning of
  // the buffer. If this token is after the previous token, start from the
  // previous token.
  size_t line = 1;
  size_t start = 0;
  if (lastLineNumberOffset > 0 && tokOffset >= lastLineNumberOffset) {
    start = lastLineNumberOffset;
    line = lastLineNumber;
  }

  line += s.substr(start, tokOffset - start).count('\n');

  // Store the line number of this token for reuse.
  lastLineNumberOffset = tokOffset;
  lastLineNumber = line;

  return line;
}

// Returns 0-based column number of the current token.
size_t ScriptLexer::getColumnNumber() {
  return prevTok.data() - getLine().data();
}

std::string ScriptLexer::getCurrentLocation() {
  std::string filename = std::string(getCurrentMB().getBufferIdentifier());
  return (filename + ":" + Twine(getLineNumber())).str();
}

ScriptLexer::ScriptLexer(MemoryBufferRef mb) {
  cur.s = mb.getBuffer();
  cur.filename = mb.getBufferIdentifier();
  mbs.push_back(mb);
}

// We don't want to record cascading errors. Keep only the first one.
void ScriptLexer::setError(const Twine &msg) {
  if (errorCount())
    return;

  std::string s = (getCurrentLocation() + ": " + msg).str();
  if (prevTok.size())
    s += "\n>>> " + getLine().str() + "\n>>> " +
         std::string(getColumnNumber(), ' ') + "^";
  error(s);
}

void ScriptLexer::lexToken() {
  std::vector<StringRef> vec;
  StringRef begin = cur.s;

  for (;;) {
    cur.s = skipSpace(cur.s);
    if (cur.s.empty()) {
      // If this buffer is from an INCLUDE command, switch to the "return
      // value"; otherwise, mark EOF.
      if (buffers.empty()) {
        eof = true;
        return;
      }
      cur = buffers.pop_back_val();
      continue;
    }
    curTokState = inExpr;

    // Quoted token. Note that double-quote characters are parts of a token
    // because, in a glob match context, only unquoted tokens are interpreted
    // as glob patterns. Double-quoted tokens are literal patterns in that
    // context.
    if (cur.s.starts_with("\"")) {
      size_t e = cur.s.find("\"", 1);
      if (e == StringRef::npos) {
        size_t lineno =
            begin.substr(0, cur.s.data() - begin.data()).count('\n');
        error(cur.filename + ":" + Twine(lineno + 1) + ": unclosed quote");
        return;
      }

      curTok = cur.s.take_front(e + 1);
      cur.s = cur.s.substr(e + 1);
      return;
    }

    // Some operators form separate tokens.
    if (cur.s.starts_with("<<=") || cur.s.starts_with(">>=")) {
      curTok = cur.s.substr(0, 3);
      cur.s = cur.s.substr(3);
      return;
    }
    if (cur.s.size() > 1 &&
        ((cur.s[1] == '=' && strchr("*/+-!<>&^|", cur.s[0])) ||
         (cur.s[0] == cur.s[1] && strchr("<>&|", cur.s[0])))) {
      curTok = cur.s.substr(0, 2);
      cur.s = cur.s.substr(2);
      return;
    }

    // Unquoted token. The non-expression token is more relaxed than tokens in
    // C-like languages, so that you can write "file-name.cpp" as one bare
    // token.
    size_t pos;
    if (inExpr) {
      pos = cur.s.find_first_not_of(
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
          "0123456789_.$");
      if (pos == 0 && cur.s.size() >= 2 &&
          is_contained({"==", "!=", "<=", ">=", "<<", ">>"},
                       cur.s.substr(0, 2)))
        pos = 2;
    } else {
      pos = cur.s.find_first_not_of(
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
          "0123456789_.$/\\~=+[]*?-!^:");
    }

    if (pos == 0)
      pos = 1;
    curTok = cur.s.substr(0, pos);
    cur.s = cur.s.substr(pos);
    break;
  }
}

// Skip leading whitespace characters or comments.
StringRef ScriptLexer::skipSpace(StringRef s) {
  for (;;) {
    if (s.starts_with("/*")) {
      size_t e = s.find("*/", 2);
      if (e == StringRef::npos) {
        setError("unclosed comment in a linker script");
        return "";
      }
      s = s.substr(e + 2);
      continue;
    }
    if (s.starts_with("#")) {
      size_t e = s.find('\n', 1);
      if (e == StringRef::npos)
        e = s.size() - 1;
      s = s.substr(e + 1);
      continue;
    }
    size_t size = s.size();
    s = s.ltrim();
    if (s.size() == size)
      return s;
  }
}

// An erroneous token is handled as if it were the last token before EOF.
bool ScriptLexer::atEOF() { return eof || errorCount(); }

StringRef ScriptLexer::next() {
  if (errorCount())
    return "";
  prevTok = peek();
  return std::exchange(curTok, StringRef(cur.s.data(), 0));
}

StringRef ScriptLexer::peek() {
  // curTok is invalid if curTokState and inExpr mismatch.
  if (curTok.size() && curTokState != inExpr) {
    cur.s = StringRef(curTok.data(), cur.s.end() - curTok.data());
    curTok = {};
  }
  if (curTok.empty())
    lexToken();
  return curTok;
}

bool ScriptLexer::consume(StringRef tok) {
  if (peek() != tok)
    return false;
  next();
  return true;
}

void ScriptLexer::skip() { (void)next(); }

void ScriptLexer::expect(StringRef expect) {
  if (errorCount())
    return;
  StringRef tok = next();
  if (tok != expect) {
    if (atEOF())
      setError("unexpected EOF");
    else
      setError(expect + " expected, but got " + tok);
  }
}

// Returns true if S encloses T.
static bool encloses(StringRef s, StringRef t) {
  return s.bytes_begin() <= t.bytes_begin() && t.bytes_end() <= s.bytes_end();
}

MemoryBufferRef ScriptLexer::getCurrentMB() {
  // Find input buffer containing the current token.
  assert(!mbs.empty());
  if (prevTok.empty())
    return mbs.back();
  for (MemoryBufferRef mb : mbs)
    if (encloses(mb.getBuffer(), cur.s))
      return mb;
  llvm_unreachable("getCurrentMB: failed to find a token");
}
