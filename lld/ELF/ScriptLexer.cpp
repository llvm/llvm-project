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
#include "llvm/ADT/StringMap.h"
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

  S += itBegin->val;
  while (++itBegin != itEnd) {
    S += " ";
    S += itBegin->val;
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

      vec.push_back({Tok::Quote, s.take_front(e + 1)});
      s = s.substr(e + 1);
      continue;
    }
    // Some operators form separate tokens.
    if (s.starts_with("<<=") || s.starts_with(">>=")) {
      vec.push_back(getOperatorToken(s));
      s = s.substr(3);
      continue;
    }
    if (s.size() > 1 && ((s[1] == '=' && strchr("*/+-<>&^|", s[0])) ||
                         (s[0] == s[1] && strchr("<>&|", s[0])))) {
      vec.push_back(getOperatorToken(s));
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
    if (pos == 0) {
      pos = 1;
      vec.push_back(getOperatorToken(s));
    } else {
      vec.push_back(getKeywordorIdentifier(s.substr(0, pos)));
    }
    s = s.substr(pos);
  }

  tokens.insert(tokens.begin() + pos, vec.begin(), vec.end());
}

ScriptLexer::Token ScriptLexer::getOperatorToken(StringRef s) {
  auto createToken = [&](Tok kind, size_t pos) -> Token {
    return {kind, s.substr(0, pos)};
  };

  switch (s.front()) {
  case EOF:
    return createToken(Tok::Eof, 0);
  case '(':
    return createToken(Tok::BracektBegin, 1);
  case ')':
    return createToken(Tok::BracektEnd, 1);
  case '{':
    return createToken(Tok::CurlyBegin, 1);
  case '}':
    return createToken(Tok::CurlyEnd, 1);
  case ';':
    return createToken(Tok::Semicolon, 1);
  case ',':
    return createToken(Tok::Comma, 1);
  case ':':
    return createToken(Tok::Colon, 1);
  case '?':
    return createToken(Tok::Question, 1);
  case '%':
    return createToken(Tok::Percent, 1);
  case '!':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::NotEqual, 2);
    return createToken(Tok::Excalamation, 1);
  case '*':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::MulAssign, 2);
    return createToken(Tok::Asterisk, 1);
  case '/':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::DivAssign, 2);
    return createToken(Tok::Slash, 1);
  case '=':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::Equal, 2);
    return createToken(Tok::Assign, 1);
  case '+':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::PlusAssign, 2);
    return createToken(Tok::Plus, 1);
  case '-':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::MinusAssign, 2);
    return createToken(Tok::Minus, 1);
  case '<':
    if (s.size() > 2 && s[1] == s[0] && s[2] == '=')
      return createToken(Tok::LeftShiftAssign, 3);
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Tok::LessEqual, 2);
      if (s[1] == '<')
        return createToken(Tok::LeftShift, 2);
    }
    return createToken(Tok::Less, 1);
  case '>':
    if (s.size() > 2 && s[1] == s[0] && s[2] == '=')
      return createToken(Tok::RightShiftAssign, 3);
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Tok::GreaterEqual, 2);
      if (s[1] == '>')
        return createToken(Tok::RightShift, 2);
    }
    return createToken(Tok::Greater, 1);
  case '&':
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Tok::AndAssign, 2);
      if (s[1] == '&')
        return createToken(Tok::AndGate, 2);
    }
    return createToken(Tok::Bitwise, 1);
  case '^':
    if (s.size() > 1 && s[1] == '=')
      return createToken(Tok::XorAssign, 2);
    return createToken(Tok::Xor, 1);
  case '|':
    if (s.size() > 1) {
      if (s[1] == '=')
        return createToken(Tok::OrAssign, 2);
      if (s[1] == '|')
        return createToken(Tok::OrGate, 2);
    }
    return createToken(Tok::Or, 1);
  case '.':
    return createToken(Tok::Dot, 1);
  case '_':
    return createToken(Tok::Underscore, 1);
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return createToken(Tok::Decimal, 1);
  default:
    return {Tok::Identifier, s};
  }
}

const llvm::StringMap<Tok> ScriptLexer::keywordTokMap = {
    {"ENTRY", Tok::Entry},
    {"INPUT", Tok::Input},
    {"GROUP", Tok::Group},
    {"INCLUDE", Tok::Include},
    {"MEMORY", Tok::Memory},
    {"OUTPUT", Tok::Output},
    {"SEARCH_DIR", Tok::SearchDir},
    {"STARTUP", Tok::Startup},
    {"INSERT", Tok::Insert},
    {"AFTER", Tok::After},
    {"OUTPUT_FORMAT", Tok::OutputFormat},
    {"TARGET", Tok::Target},
    {"ASSERT", Tok::Assert},
    {"CONSTANT", Tok::Constant},
    {"EXTERN", Tok::Extern},
    {"OUTPUT_ARCH", Tok::OutputArch},
    {"NOCROSSREFS", Tok::Nocrossrefs},
    {"NOCROSSREFS_TO", Tok::NocrossrefsTo},
    {"PROVIDE", Tok::Provide},
    {"HIDDEN", Tok::Hidden},
    {"PROVIDE_HIDDEN", Tok::ProvideHidden},
    {"SECTIONS", Tok::Sections},
    {"BEFORE", Tok::Before},
    {"EXCLUDE_FILE", Tok::ExcludeFile},
    {"KEEP", Tok::Keep},
    {"INPUT_SECTION_FLAGS", Tok::InputSectionFlags},
    {"OVERLAY", Tok::Overlay},
    {"NOLOAD", Tok::Noload},
    {"COPY", Tok::Copy},
    {"INFO", Tok::Info},
    {"OVERWRITE_SECTIONS", Tok::OverwriteSections},
    {"SUBALIGN", Tok::Subalign},
    {"ONLY_IF_RO", Tok::OnlyIfRO},
    {"ONLY_IF_RW", Tok::OnlyIfRW},
    {"FILL", Tok::Fill},
    {"SORT", Tok::Sort},
    {"ABSOLUTE", Tok::Absolute},
    {"ADDR", Tok::Addr},
    {"ALIGN", Tok::Align},
    {"ALIGNOF", Tok::Alignof},
    {"DATA_SEGMENT_ALIGN", Tok::DataSegmentAlign},
    {"DATA_SEGMENT_END", Tok::DataSegmentEnd},
    {"DATA_SEGMENT_RELRO_END", Tok::DataSegmentRelroEnd},
    {"DEFINED", Tok::Defined},
    {"LENGTH", Tok::Length},
    {"LOADADDR", Tok::Loadaddr},
    {"LOG2CEIL", Tok::Log2ceil},
    {"MAX", Tok::Max},
    {"MIN", Tok::Min},
    {"ORIGIN", Tok::Origin},
    {"SEGMENT_START", Tok::SegmentStart},
    {"SIZEOF", Tok::Sizeof},
    {"SIZEOF_HEADERS", Tok::SizeofHeaders},
    {"FILEHDR", Tok::Filehdr},
    {"PHDRS", Tok::Phdrs},
    {"AT", Tok::At},
    {"FLAGS", Tok::Flags},
    {"VERSION", Tok::Version},
    {"REGION_ALIAS", Tok::RegionAlias},
    {"AS_NEEDED", Tok::AsNeeded},
    {"CONSTRUCTORS", Tok::Constructors},
    {"MAXPAGESIZE", Tok::Maxpagesize},
    {"COMMONPAGESIZE", Tok::Commonpagesize}};

ScriptLexer::Token ScriptLexer::getKeywordorIdentifier(StringRef s) {
  auto it = keywordTokMap.find(s.str());
  if (it != keywordTokMap.end())
    return {it->second, s};
  return {Tok::Identifier, s};
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
    return {{Tok::Quote, s}};

  // Split S with operators as separators.
  std::vector<ScriptLexer::Token> ret;
  while (!s.empty()) {
    size_t e = s.find_first_of(ops);

    // No need to split if there is no operator.
    if (e == StringRef::npos) {
      ret.push_back({Tok::Identifier, s});
      break;
    }

    // Get a token before the operator.
    if (e != 0)
      ret.push_back({Tok::Identifier, s.substr(0, e)});

    // Get the operator as a token.
    // Keep !=, ==, >=, <=, << and >> operators as a single tokens.
    if (s.substr(e).starts_with("!=") || s.substr(e).starts_with("==") ||
        s.substr(e).starts_with(">=") || s.substr(e).starts_with("<=") ||
        s.substr(e).starts_with("<<") || s.substr(e).starts_with(">>")) {
      ret.push_back(getOperatorToken(s.substr(e)));
      s = s.substr(e + 2);
    } else {
      ret.push_back(getOperatorToken(s.substr(e, 1)));
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
  std::vector<Token> v = tokenizeExpr(tokens[pos].val);
  if (v.size() == 1)
    return;
  tokens.erase(tokens.begin() + pos);
  tokens.insert(tokens.begin() + pos, v.begin(), v.end());
}

ScriptLexer::Token ScriptLexer::next() {
  if (errorCount())
    return {Tok::Error, ""};
  if (atEOF()) {
    setError("unexpected EOF");
    return {Tok::Eof, ""};
  }
  if (inExpr)
    maybeSplitExpr();
  return tokens[pos++];
}

ScriptLexer::Token ScriptLexer::peek() {
  Token tok = next();
  if (errorCount())
    return {Tok::Error, ""};
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
  Token tok = next();
  if (tok != expect)
    setError(expect + " expected, but got " + tok.val);
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
