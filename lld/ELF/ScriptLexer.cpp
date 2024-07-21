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
// Our grammar of the linker script is LL(2), meaning that it needs at
// most two-token lookahead to parse. The only place we need two-token
// lookahead is labels in version scripts, where we need to parse "local :"
// as if "local:".
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
  StringRef tok = tokens[pos - 1].val;

  size_t pos = s.rfind('\n', tok.data() - s.data());
  if (pos != StringRef::npos)
    s = s.substr(pos + 1);
  return s.substr(0, s.find_first_of("\r\n"));
}

// Returns 1-based line number of the current token.
size_t ScriptLexer::getLineNumber() {
  if (pos == 0)
    return 1;
  StringRef s = getCurrentMB().getBuffer();
  StringRef tok = tokens[pos - 1].val;
  const size_t tokOffset = tok.data() - s.data();

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
  StringRef tok = tokens[pos - 1].val;
  return tok.data() - getLine().data();
}

std::string ScriptLexer::getCurrentLocation() {
  std::string filename = std::string(getCurrentMB().getBufferIdentifier());
  return (filename + ":" + Twine(getLineNumber())).str();
}

std::string ScriptLexer::joinTokens(size_t begin, size_t end) {
  auto itBegin = tokens.begin() + begin;
  auto itEnd = tokens.begin() + end;

  std::string S;
  if (itBegin == itEnd)
    return S;

  S += (*itBegin).val;
  while (++itBegin != itEnd) {
    S += " ";
    S += (*itBegin).val;
  }
  return S;
}

ScriptLexer::ScriptLexer(MemoryBufferRef mb) { tokenize(mb); }

// We don't want to record cascading errors. Keep only the first one.
void ScriptLexer::setError(const Twine &msg) {
  if (errorCount())
    return;

  std::string s = (getCurrentLocation() + ": " + msg).str();
  if (pos)
    s += "\n>>> " + getLine().str() + "\n>>> " +
         std::string(getColumnNumber(), ' ') + "^";
  error(s);
}

// Split S into linker script tokens.
void ScriptLexer::tokenize(MemoryBufferRef mb) {
  std::vector<Token> vec;
  mbs.push_back(mb);
  StringRef s = mb.getBuffer();
  StringRef begin = s;

  for (;;) {
    s = skipSpace(s);
    if (s.empty())
      break;

    // Quoted token. Note that double-quote characters are parts of a token
    // because, in a glob match context, only unquoted tokens are interpreted
    // as glob patterns. Double-quoted tokens are literal patterns in that
    // context.
    if (s.starts_with("\"")) {
      size_t e = s.find("\"", 1);
      if (e == StringRef::npos) {
        StringRef filename = mb.getBufferIdentifier();
        size_t lineno = begin.substr(0, s.data() - begin.data()).count('\n');
        error(filename + ":" + Twine(lineno + 1) + ": unclosed quote");
        return;
      }

      vec.push_back({Kind::Quote, s.take_front(e + 1)});
      s = s.substr(e + 1);
      continue;
    }
    // Some operators form separate tokens.
    if (s.starts_with("<<=") || s.starts_with(">>=")) {
      vec.push_back({Kind::LeftShiftAssign, s.substr(0, 3)});
      s = s.substr(3);
      continue;
    }
    if (s.size() > 1 && ((s[1] == '=' && strchr("*/+-<>&^|", s[0])) ||
                         (s[0] == s[1] && strchr("<>&|", s[0])))) {
      vec.push_back({Kind::Assign, s.substr(0, 2)});
      s = s.substr(2);
      continue;
    }

    // Unquoted token. This is more relaxed than tokens in C-like language,
    // so that you can write "file-name.cpp" as one bare token, for example.
    size_t pos = s.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789_.$/\\~=+[]*?-!^:");

    // A character that cannot start a word (which is usually a
    // punctuation) forms a single character token.
    if (pos == 0)
      pos = 1;
    vec.push_back({Kind::Identifier, s.substr(0, pos)});
    s = s.substr(pos);
  }

  tokens.insert(tokens.begin() + pos, vec.begin(), vec.end());
}

ScriptLexer::Token ScriptLexer::getOperatorToken(StringRef s) {
  auto createToken = [&](Kind kind, size_t pos) -> Token {
    return {kind, s.substr(0, pos)};
  };

  switch (s.front()) {
  case EOF:
    return createToken(Kind::Eof, 1);
  case '(':
    return createToken(Kind::BracektBegin, 1);
  case ')':
    return createToken(Kind::BracektEnd, 1);
  case '{':
    return createToken(Kind::CurlyBegin, 1);
  case '}':
    return createToken(Kind::CurlyEnd, 1);
  case ';':
    return createToken(Kind::Semicolon, 1);
  case ',':
    return createToken(Kind::Comma, 1);
  case ':':
    return createToken(Kind::Colon, 1);
  case '?':
    return createToken(Kind::Question, 1);
  case '!':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::NotEqual, 2);
    return createToken(Kind::Excalamation, 1);
  case '*':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::MulAssign, 2);
    return createToken(Kind::Asterisk, 1);
  case '/':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::DivAssign, 2);
    return createToken(Kind::Slash, 1);
  case '=':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::Equal, 2);
    return createToken(Kind::Assign, 1);
  case '+':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::PlusAssign, 2);
    else
      return createToken(Kind::Plus, 1);
  case '-':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::MinusAssign, 2);
    return createToken(Kind::Minus, 1);
  case '<':
    if (s.size() > 2 && s[1] == s[0] && s[2] == '=')
      return createToken(Kind::LeftShiftAssign, 3);
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Kind::LessEqual, 2);
      if (s[1] == '<')
        return createToken(Kind::LeftShift, 2);
    }
    return createToken(Kind::Less, 1);
  case '>':
    if (s.size() > 2 && s[1] == s[0] && s[2] == '=')
      return createToken(Kind::RightShiftAssign, 3);
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Kind::GreaterEqual, 2);
      if (s[1] == '>')
        return createToken(Kind::RightShift, 2);
    }
    return createToken(Kind::Greater, 1);
  case '&':
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Kind::AndAssign, 2);
      if (s[1] == '&')
        return createToken(Kind::AndGate, 2);
    }
    return createToken(Kind::Bitwise, 1);
  case '^':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Kind::XorAssign, 2);
    return createToken(Kind::Xor, 1);
  case '|':
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Kind::OrAssign, 2);
      if (s[1] == '|')
        return createToken(Kind::OrGate, 2);
    }
    return createToken(Kind::Or, 1);
  case '.':
  case '_':
    // TODO
  default:
    return createToken(Kind::Error, 1);
  }
}

ScriptLexer::Token ScriptLexer::getKeywordorIdentifier(StringRef s) {
  static const std::unordered_map<std::string, Kind> keywords = {
      {"ENTRY", Kind::Entry},
      {"INPUT", Kind::Input},
      {"GROUP", Kind::Group},
      {"MEMORY", Kind::Memory},
      {"OUTPUT", Kind::Output},
      {"SEARCH_DIR", Kind::SearchDir},
      {"STARTUP", Kind::Startup},
      {"INSERT", Kind::Insert},
      {"AFTER", Kind::After},
      {"OUTPUT_FORMAT", Kind::OutputFormat},
      {"TARGET", Kind::Target},
      {"ASSERT", Kind::Assert},
      {"CONSTANT", Kind::Constant},
      {"EXTERN", Kind::Extern},
      {"OUTPUT_ARCH", Kind::OutputArch},
      {"NOCROSSREFS", Kind::Nocrossrefs},
      {"NOCROSSREFS_TO", Kind::NocrossrefsTo},
      {"PROVIDE", Kind::Provide},
      {"HIDDEN", Kind::Hidden},
      {"PROVIDE_HIDDEN", Kind::ProvideHidden},
      {"SECTIONS", Kind::Sections},
      {"BEFORE", Kind::Before},
      {"EXCLUDE_FILE", Kind::ExcludeFile},
      {"KEEP", Kind::Keep},
      {"INPUT_SECTION_FLAGS", Kind::InputSectionFlags},
      {"OVERLAY", Kind::Overlay},
      {"Noload", Kind::Noload},
      {"COPY", Kind::Copy},
      {"INFO", Kind::Info},
      {"OVERWRITE_SECTIONS", Kind::OverwriteSections},
      {"SUBALIGN", Kind::Subalign},
      {"ONLY_IF_RO", Kind::OnlyIfRO},
      {"ONLY_IF_RW", Kind::OnlyIfRW},
      {"FILL", Kind::Fill},
      {"SORT", Kind::Sort},
      {"ABSOLUTE", Kind::Absolute},
      {"ADDR", Kind::Addr},
      {"ALIGN", Kind::Align},
      {"ALIGNOF", Kind::Alignof},
      {"DATA_SEGMENT_ALIGN", Kind::DataSegmentAlign},
      {"DATA_SEGMENT_END", Kind::DataSegmentEnd},
      {"DEFINED", Kind::Defined},
      {"LENGTH", Kind::Length},
      {"LOADADDR", Kind::Loadaddr},
      {"LOG2CEIL", Kind::Log2ceil},
      {"MAX", Kind::Max},
      {"MIN", Kind::Min},
      {"ORIGIN", Kind::Origin},
      {"SEGMENT_START", Kind::SegmentStart},
      {"SIZEOF", Kind::Sizeof},
      {"SIZEOF_HEADERS", Kind::SizeofHeaders},
      {"FILEHDR", Kind::Filehdr},
      {"PHDRS", Kind::Phdrs},
      {"AT", Kind::At},
      {"FLAGS", Kind::Flags},
      {"VERSION", Kind::Version},
      {"REGION_ALIAS", Kind::RegionAlias},
      {"AS_NEEDED", Kind::AsNeeded},
      {"CONSTRUCTORS", Kind::Constructors},
      {"MAXPAGESIZE", Kind::Maxpagesize},
      {"COMMONPAGESIZE", Kind::Commonpagesize}};
  auto it = keywords.find(s.str());
  if (it != keywords.end())
    return {it->second, s};
  return {Kind::Identifier, s};
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
bool ScriptLexer::atEOF() { return errorCount() || tokens.size() == pos; }

// Split a given string as an expression.
// This function returns "3", "*" and "5" for "3*5" for example.
std::vector<ScriptLexer::Token> ScriptLexer::tokenizeExpr(StringRef s) {
  StringRef ops = "!~*/+-<>?^:="; // List of operators

  // Quoted strings are literal strings, so we don't want to split it.
  if (s.starts_with("\""))
    return {{Kind::Quote, s}};

  // Split S with operators as separators.
  std::vector<ScriptLexer::Token> ret;
  while (!s.empty()) {
    size_t e = s.find_first_of(ops);

    // No need to split if there is no operator.
    if (e == StringRef::npos) {
      ret.push_back({Kind::Identifier, s});
      break;
    }

    // Get a token before the operator.
    if (e != 0)
      ret.push_back({Kind::Identifier, s.substr(0, e)});

    // Get the operator as a token.
    // Keep !=, ==, >=, <=, << and >> operators as a single tokens.
    if (s.substr(e).starts_with("!=") || s.substr(e).starts_with("==") ||
        s.substr(e).starts_with(">=") || s.substr(e).starts_with("<=") ||
        s.substr(e).starts_with("<<") || s.substr(e).starts_with(">>")) {
      ret.push_back({Kind::GreaterEqual, s.substr(e, 2)});
      s = s.substr(e + 2);
    } else {
      ret.push_back({Kind::Identifier, s.substr(e, 1)});
      s = s.substr(e + 1);
    }
  }
  return ret;
}

// In contexts where expressions are expected, the lexer should apply
// different tokenization rules than the default one. By default,
// arithmetic operator characters are regular characters, but in the
// expression context, they should be independent tokens.
//
// For example, "foo*3" should be tokenized to "foo", "*" and "3" only
// in the expression context.
//
// This function may split the current token into multiple tokens.
void ScriptLexer::maybeSplitExpr() {
  if (!inExpr || errorCount() || atEOF())
    return;

  std::vector<Token> v = tokenizeExpr(tokens[pos].val);
  if (v.size() == 1)
    return;
  tokens.erase(tokens.begin() + pos);
  tokens.insert(tokens.begin() + pos, v.begin(), v.end());
}

StringRef ScriptLexer::next() {
  maybeSplitExpr();

  if (errorCount())
    return "";
  if (atEOF()) {
    setError("unexpected EOF");
    return "";
  }
  return tokens[pos++].val;
}

StringRef ScriptLexer::peek() {
  StringRef tok = next();
  if (errorCount())
    return "";
  pos = pos - 1;
  return tok;
}

bool ScriptLexer::consume(StringRef tok) {
  if (next() == tok)
    return true;
  --pos;
  return false;
}

// Consumes Tok followed by ":". Space is allowed between Tok and ":".
bool ScriptLexer::consumeLabel(StringRef tok) {
  if (consume((tok + ":").str()))
    return true;
  if (tokens.size() >= pos + 2 && tokens[pos].val == tok &&
      tokens[pos + 1].val == ":") {
    pos += 2;
    return true;
  }
  return false;
}

void ScriptLexer::skip() { (void)next(); }

void ScriptLexer::expect(StringRef expect) {
  if (errorCount())
    return;
  StringRef tok = next();
  if (tok != expect)
    setError(expect + " expected, but got " + tok);
}

// Returns true if S encloses T.
static bool encloses(StringRef s, StringRef t) {
  return s.bytes_begin() <= t.bytes_begin() && t.bytes_end() <= s.bytes_end();
}

MemoryBufferRef ScriptLexer::getCurrentMB() {
  // Find input buffer containing the current token.
  assert(!mbs.empty());
  if (pos == 0)
    return mbs.back();
  for (MemoryBufferRef mb : mbs)
    if (encloses(mb.getBuffer(), tokens[pos - 1].val))
      return mb;
  llvm_unreachable("getCurrentMB: failed to find a token");
}
