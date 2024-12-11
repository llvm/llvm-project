//===-- DILLexer.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILLEXER_H_
#define LLDB_VALUEOBJECT_DILLEXER_H_

#include "llvm/TargetParser/Host.h"

#include <cstdint>
#include <memory>
#include <limits.h>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace lldb_private {

namespace dil {

enum class TokenKind {
  amp,
  ampamp,
  ampequal,
  arrow,
  caret,
  caretequal,
  char_constant,
  colon,
  coloncolon,
  comma,
  equal,
  equalequal,
  eof,
  exclaim,
  exclaimequal,
  flt,
  greater,
  greaterequal,
  greatergreater,
  greatergreaterequal,
  identifier,
  integer,
  invalid,
  kw_bool,
  kw_char,
  kw_double,
  kw_dynamic_cast,
  kw_false,
  kw_float,
  kw_int,
  kw_long,
  kw_nullptr,
  kw_reinterpret_cast,
  kw_short,
  kw_signed,
  kw_static_cast,
  kw_this,
  kw_true,
  kw_unsigned,
  l_paren,
  l_square,
  less,
  lessequal,
  lessless,
  lesslessequal,
  minus,
  minusequal,
  minusminus,
  none,
  numeric_constant,
  percent,
  percentequal,
  period,
  pipe,
  pipeequal,
  pipepipe,
  plus,
  plusequal,
  plusplus,
  question,
  r_paren,
  r_square,
  slash,
  slashequal,
  star,
  starequal,
  string_literal,
  tilde,
  unknown,
  utf8_char_constant,
  utf8_string_literal,
  wide_char_constant,
  wide_string_literal,
  word,
  // type keywords -- Lexer does not recognize these yet.
  kw_char16_t,
  kw_char32_t,
  kw_const,
  kw_namespace,
  kw_sizeof,
  kw_void,
  kw_volatile,
  kw_wchar_t,
};

enum class TypeSpecifier {
  kBool,
  kChar,
  kChar16,
  kChar32,
  kDouble,
  kFloat,
  kInt,
  kLong,
  kLongDouble,
  kLongLong,
  kShort,
  kUnknown,
  kVoid,
  kWChar,
};

enum class SignSpecifier {
  eSigned,
  eUnsigned,
};

enum class NumberKind {
  eInteger,
  eFloat,
  eNone
};

class DILToken {
 public:
  DILToken (dil::TokenKind kind, std::string spelling, uint32_t start,
            uint32_t len) :
      m_kind(kind), m_spelling(spelling), m_start_pos(start), m_length(len) {}

  DILToken () :
      m_kind(dil::TokenKind::none), m_spelling(""), m_start_pos(0),
      m_length(0) {}

  void setKind(dil::TokenKind kind) { m_kind = kind; }
  dil::TokenKind getKind() const { return m_kind; }

  std::string getSpelling() const { return m_spelling; }

  uint32_t getLength() const { return m_length; }

  bool is (dil::TokenKind kind) const { return m_kind == kind; }

  bool isNot(dil::TokenKind kind) const { return m_kind != kind; }

  bool isOneOf(dil::TokenKind kind1, dil::TokenKind kind2) const {
    return is(kind1) || is(kind2);
  }

  template <typename... Ts> bool isOneOf(dil::TokenKind kind, Ts... Ks) const {
    return is(kind) || isOneOf(Ks...);
  }

  uint32_t getLocation() const { return m_start_pos; }

  void setValues (dil::TokenKind kind, std::string spelling,
                  uint32_t start, uint32_t len) {
    m_kind = kind;
    m_spelling = spelling;
    m_start_pos = start;
    m_length = len;
  }

  static const std::string getTokenName(dil::TokenKind kind);

 private:
  dil::TokenKind m_kind;
  std::string m_spelling;
  uint32_t m_start_pos; // within entire expression string
  uint32_t m_length;
};

class DILSourceManager {
 public:
  static std::shared_ptr<DILSourceManager> Create(std::string expr);

  // This class cannot be safely moved because of the dependency between
  // `m_expr` and `m_smff`. Users are supposed to pass around the shared
  // pointer.
  DILSourceManager(DILSourceManager&&) = delete;
  DILSourceManager(const DILSourceManager&) = delete;
  DILSourceManager& operator=(DILSourceManager const&) = delete;

  std::string GetSource() { return m_expr; }

 private:
  explicit DILSourceManager(std::string expr)  : m_expr(std::move(expr)) {}

 private:
  // Store the expression, since SourceManagerForFile doesn't take the
  // ownership.
  std::string m_expr;
};


class DILLexer {
 public:

  DILLexer(std::shared_ptr<DILSourceManager> dil_sm) :
      m_expr(dil_sm->GetSource()) {
    m_cur_pos = m_expr.begin();
    // Use UINT_MAX to indicate invalid/uninitialized value.
    m_tokens_idx = UINT_MAX;
  }

  bool Lex(DILToken &result, bool look_ahead=false);

  bool Is_Word(std::string::iterator start, uint32_t& length);

  bool Is_Number(std::string::iterator start, uint32_t& length,
                 dil::NumberKind& kind);

  bool isStringLiteral(dil::TokenKind kind) {
    return (kind == dil::TokenKind::string_literal ||
            kind == dil::TokenKind::wide_string_literal ||
            kind == dil::TokenKind::utf8_string_literal);
  }

  uint32_t GetLocation() { return m_cur_pos - m_expr.begin(); }

  /// Update 'result' with the other paremeter values, create a
  /// duplicate token, and push the duplicate token onto the vector of
  /// lexed tokens.
  void UpdateLexedTokens (DILToken &result, dil::TokenKind tok_kind,
                          std::string tok_str, uint32_t tok_pos,
                          uint32_t tok_len);

  /// Return the lexed token N+1 positions ahead of the 'current' token
  /// being handled by the DIL parser.
  const DILToken &LookAhead(uint32_t N);

  const DILToken &AcceptLookAhead(uint32_t N);

  /// Return the index for the 'current' token being handled by the DIL parser.
  uint32_t GetCurrentTokenIdx() { return m_tokens_idx; }

  /// Return the current token to be handled by the DIL parser.
  DILToken& GetCurrentToken() { return m_lexed_tokens[m_tokens_idx]; }

  /// Update the index for the 'current' token, to point to the next lexed
  /// token.
  bool IncrementTokenIdx() {
    if (m_tokens_idx >= m_lexed_tokens.size() - 1)
      return false;

    m_tokens_idx++;
    return true;
  }

  /// Set the index for the 'current' token (to be handled by the parser)
  /// to a particular position. Used for either committing 'look ahead' parsing
  /// or rolling back tentative parsing.
  bool ResetTokenIdx(uint32_t new_value) {
    if (new_value > m_lexed_tokens.size() - 1)
      return false;

    m_tokens_idx = new_value;
    return true;
  }

  uint32_t getCharWidth() { return 8; }
  uint32_t getIntWidth() { return 32; }
  uint32_t getWCharWidth() { return 16; }


 private:
  // The input string we are lexing & parsing.
  std::string m_expr;

  // The current position of the lexer within m_expr (the character position,
  // within the string, of the next item to be lexed).
  std::string::iterator m_cur_pos;

  // Holds all of the tokens lexed so far.
  std::vector<DILToken> m_lexed_tokens;

  // Index into m_lexed_tokens; indicates which token the DIL parser is
  // currently trying to parse/handle.
  uint32_t m_tokens_idx;

  // "invalid" token; to be returned by lexer when 'look ahead' fails.
  DILToken m_invalid_token;
};

} // namespace dil

} // namespace lldb_private


#endif // LLDB_VALUEOBJECT_DILLEXER_H_
