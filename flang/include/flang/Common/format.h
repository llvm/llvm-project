//===-- include/flang/Common/format.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_FORMAT_H_
#define FORTRAN_COMMON_FORMAT_H_

#include "enum-set.h"
#include "flang/Common/Fortran.h"
#include <cstring>

// Define a FormatValidator class template to validate a format expression
// of a given CHAR type.  To enable use in runtime library code as well as
// compiler code, the implementation does its own parsing without recourse
// to compiler parser machinery, and avoids features that require C++ runtime
// library support.  A format expression is a pointer to a fixed size
// character string, with an explicit length.  Class function Check analyzes
// the expression for syntax and semantic errors and warnings.  When an error
// or warning is found, a caller-supplied reporter function is called, which
// may request early termination of validation analysis when some threshold
// number of errors have been reported.  If the context is a READ, WRITE,
// or PRINT statement, rather than a FORMAT statement, statement-specific
// checks are also done.

namespace Fortran::common {

struct FormatMessage {
  const char *text; // message text; may have one %s argument
  const char *arg; // optional %s argument value
  int offset; // offset to message marker
  int length; // length of message marker
  bool isError; // vs. warning
};

// This declaration is logically private to class FormatValidator.
// It is placed here to work around a clang compilation problem.
ENUM_CLASS(TokenKind, None, A, B, BN, BZ, D, DC, DP, DT, E, EN, ES, EX, F, G, I,
    L, O, P, RC, RD, RN, RP, RU, RZ, S, SP, SS, T, TL, TR, X, Z, Colon, Slash,
    Backslash, // nonstandard: inhibit newline on output
    Dollar, // nonstandard: inhibit newline on output on terminals
    Star, LParen, RParen, Comma, Point, Sign,
    UnsignedInteger, // value in integerValue_
    String) // char-literal-constant or Hollerith constant

template <typename CHAR = char> class FormatValidator {
public:
  using Reporter = std::function<bool(const FormatMessage &)>;
  FormatValidator(const CHAR *format, size_t length, Reporter reporter,
      IoStmtKind stmt = IoStmtKind::None)
      : format_{format}, end_{format + length}, reporter_{reporter},
        stmt_{stmt}, cursor_{format - 1} {
    CHECK(format);
  }

  bool Check();
  int maxNesting() const { return maxNesting_; }

private:
  common::EnumSet<TokenKind, TokenKind_enumSize> itemsWithLeadingInts_{
      TokenKind::A, TokenKind::B, TokenKind::D, TokenKind::DT, TokenKind::E,
      TokenKind::EN, TokenKind::ES, TokenKind::EX, TokenKind::F, TokenKind::G,
      TokenKind::I, TokenKind::L, TokenKind::O, TokenKind::P, TokenKind::X,
      TokenKind::Z, TokenKind::Slash, TokenKind::LParen};

  struct Token {
    Token &set_kind(TokenKind kind) {
      kind_ = kind;
      return *this;
    }
    Token &set_offset(int offset) {
      offset_ = offset;
      return *this;
    }
    Token &set_length(int length) {
      length_ = length;
      return *this;
    }

    TokenKind kind() const { return kind_; }
    int offset() const { return offset_; }
    int length() const { return length_; }

    bool IsSet() { return kind_ != TokenKind::None; }

  private:
    TokenKind kind_{TokenKind::None};
    int offset_{0};
    int length_{1};
  };

  void ReportWarning(const char *text) { ReportWarning(text, token_); }
  void ReportWarning(
      const char *text, Token &token, const char *arg = nullptr) {
    FormatMessage msg{
        text, arg ? arg : argString_, token.offset(), token.length(), false};
    reporterExit_ |= reporter_(msg);
  }

  void ReportError(const char *text) { ReportError(text, token_); }
  void ReportError(const char *text, Token &token, const char *arg = nullptr) {
    if (suppressMessageCascade_) {
      return;
    }
    formatHasErrors_ = true;
    suppressMessageCascade_ = true;
    FormatMessage msg{
        text, arg ? arg : argString_, token.offset(), token.length(), true};
    reporterExit_ |= reporter_(msg);
  }

  void SetLength() { SetLength(token_); }
  void SetLength(Token &token) {
    token.set_length(cursor_ - format_ - token.offset() + (cursor_ < end_));
  }

  CHAR NextChar();
  CHAR LookAheadChar();
  void Advance(TokenKind);
  void NextToken();

  void check_r(bool allowed = true);
  bool check_w();
  void check_m();
  bool check_d();
  void check_e();

  const CHAR *const format_; // format text
  const CHAR *const end_; // one-past-last of format_ text
  Reporter reporter_;
  IoStmtKind stmt_;

  const CHAR *cursor_{}; // current location in format_
  const CHAR *laCursor_{}; // lookahead cursor
  Token token_{}; // current token
  TokenKind previousTokenKind_{TokenKind::None};
  int64_t integerValue_{-1}; // value of UnsignedInteger token
  Token knrToken_{}; // k, n, or r UnsignedInteger token
  int64_t knrValue_{-1}; // -1 ==> not present
  int64_t wValue_{-1};
  char argString_[3]{}; // 1-2 character msg arg; usually edit descriptor name
  bool formatHasErrors_{false};
  bool unterminatedFormatError_{false};
  bool suppressMessageCascade_{false};
  bool reporterExit_{false};
  int maxNesting_{0}; // max level of nested parentheses
};

template <typename CHAR> static inline bool IsWhite(CHAR c) {
  // White space.  ' ' is standard.  Other characters are extensions.
  // Extension candidates:
  //   '\t' (horizontal tab)
  //   '\n' (new line)
  //   '\v' (vertical tab)
  //   '\f' (form feed)
  //   '\r' (carriage ret)
  return c == ' ' || c == '\t' || c == '\v';
}

template <typename CHAR> CHAR FormatValidator<CHAR>::NextChar() {
  for (++cursor_; cursor_ < end_; ++cursor_) {
    if (!IsWhite(*cursor_)) {
      return toupper(*cursor_);
    }
  }
  cursor_ = end_; // don't allow cursor_ > end_
  return ' ';
}

template <typename CHAR> CHAR FormatValidator<CHAR>::LookAheadChar() {
  for (laCursor_ = cursor_ + 1; laCursor_ < end_; ++laCursor_) {
    if (!IsWhite(*laCursor_)) {
      return toupper(*laCursor_);
    }
  }
  laCursor_ = end_; // don't allow laCursor_ > end_
  return ' ';
}

// After a call to LookAheadChar, set token kind and advance cursor to laCursor.
template <typename CHAR> void FormatValidator<CHAR>::Advance(TokenKind tk) {
  cursor_ = laCursor_;
  token_.set_kind(tk);
}

template <typename CHAR> void FormatValidator<CHAR>::NextToken() {
  // At entry, cursor_ points before the start of the next token.
  // At exit, cursor_ points to last CHAR of token_.

  previousTokenKind_ = token_.kind();
  CHAR c{NextChar()};
  token_.set_kind(TokenKind::None);
  token_.set_offset(cursor_ - format_);
  token_.set_length(1);
  if (c == '_' && integerValue_ >= 0) { // C1305, C1309, C1310, C1312, C1313
    ReportError("Kind parameter '_' character in format expression");
  }
  integerValue_ = -1;

  switch (c) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9': {
    int64_t lastValue;
    const CHAR *lastCursor;
    integerValue_ = 0;
    bool overflow{false};
    do {
      lastValue = integerValue_;
      lastCursor = cursor_;
      integerValue_ = 10 * integerValue_ + c - '0';
      if (lastValue > integerValue_) {
        overflow = true;
      }
      c = NextChar();
    } while (c >= '0' && c <= '9');
    cursor_ = lastCursor;
    token_.set_kind(TokenKind::UnsignedInteger);
    if (overflow) {
      SetLength();
      ReportError("Integer overflow in format expression");
      break;
    }
    if (LookAheadChar() != 'H') {
      break;
    }
    // Hollerith constant
    if (laCursor_ + integerValue_ < end_) {
      token_.set_kind(TokenKind::String);
      cursor_ = laCursor_ + integerValue_;
    } else {
      token_.set_kind(TokenKind::None);
      cursor_ = end_;
    }
    SetLength();
    if (stmt_ == IoStmtKind::Read) { // 13.3.2p6
      ReportError("'H' edit descriptor in READ format expression");
    } else if (token_.kind() == TokenKind::None) {
      ReportError("Unterminated 'H' edit descriptor");
    } else {
      ReportWarning("Legacy 'H' edit descriptor");
    }
    break;
  }
  case 'A':
    token_.set_kind(TokenKind::A);
    break;
  case 'B':
    switch (LookAheadChar()) {
    case 'N':
      Advance(TokenKind::BN);
      break;
    case 'Z':
      Advance(TokenKind::BZ);
      break;
    default:
      token_.set_kind(TokenKind::B);
      break;
    }
    break;
  case 'D':
    switch (LookAheadChar()) {
    case 'C':
      Advance(TokenKind::DC);
      break;
    case 'P':
      Advance(TokenKind::DP);
      break;
    case 'T':
      Advance(TokenKind::DT);
      break;
    default:
      token_.set_kind(TokenKind::D);
      break;
    }
    break;
  case 'E':
    switch (LookAheadChar()) {
    case 'N':
      Advance(TokenKind::EN);
      break;
    case 'S':
      Advance(TokenKind::ES);
      break;
    case 'X':
      Advance(TokenKind::EX);
      break;
    default:
      token_.set_kind(TokenKind::E);
      break;
    }
    break;
  case 'F':
    token_.set_kind(TokenKind::F);
    break;
  case 'G':
    token_.set_kind(TokenKind::G);
    break;
  case 'I':
    token_.set_kind(TokenKind::I);
    break;
  case 'L':
    token_.set_kind(TokenKind::L);
    break;
  case 'O':
    token_.set_kind(TokenKind::O);
    break;
  case 'P':
    token_.set_kind(TokenKind::P);
    break;
  case 'R':
    switch (LookAheadChar()) {
    case 'C':
      Advance(TokenKind::RC);
      break;
    case 'D':
      Advance(TokenKind::RD);
      break;
    case 'N':
      Advance(TokenKind::RN);
      break;
    case 'P':
      Advance(TokenKind::RP);
      break;
    case 'U':
      Advance(TokenKind::RU);
      break;
    case 'Z':
      Advance(TokenKind::RZ);
      break;
    default:
      token_.set_kind(TokenKind::None);
      break;
    }
    break;
  case 'S':
    switch (LookAheadChar()) {
    case 'P':
      Advance(TokenKind::SP);
      break;
    case 'S':
      Advance(TokenKind::SS);
      break;
    default:
      token_.set_kind(TokenKind::S);
      break;
    }
    break;
  case 'T':
    switch (LookAheadChar()) {
    case 'L':
      Advance(TokenKind::TL);
      break;
    case 'R':
      Advance(TokenKind::TR);
      break;
    default:
      token_.set_kind(TokenKind::T);
      break;
    }
    break;
  case 'X':
    token_.set_kind(TokenKind::X);
    break;
  case 'Z':
    token_.set_kind(TokenKind::Z);
    break;
  case '-':
  case '+':
    token_.set_kind(TokenKind::Sign);
    break;
  case '/':
    token_.set_kind(TokenKind::Slash);
    break;
  case '(':
    token_.set_kind(TokenKind::LParen);
    break;
  case ')':
    token_.set_kind(TokenKind::RParen);
    break;
  case '.':
    token_.set_kind(TokenKind::Point);
    break;
  case ':':
    token_.set_kind(TokenKind::Colon);
    break;
  case '\\':
    token_.set_kind(TokenKind::Backslash);
    break;
  case '$':
    token_.set_kind(TokenKind::Dollar);
    break;
  case '*':
    token_.set_kind(LookAheadChar() == '(' ? TokenKind::Star : TokenKind::None);
    break;
  case ',': {
    token_.set_kind(TokenKind::Comma);
    CHAR laChar = LookAheadChar();
    if (laChar == ',') {
      Advance(TokenKind::Comma);
      token_.set_offset(cursor_ - format_);
      ReportError("Unexpected ',' in format expression");
    } else if (laChar == ')') {
      ReportError("Unexpected ',' before ')' in format expression");
    }
    break;
  }
  case '\'':
  case '"':
    for (++cursor_; cursor_ < end_; ++cursor_) {
      if (*cursor_ == c) {
        if (auto nc{cursor_ + 1}; nc < end_ && *nc != c) {
          token_.set_kind(TokenKind::String);
          break;
        }
        ++cursor_;
      }
    }
    SetLength();
    if (stmt_ == IoStmtKind::Read &&
        previousTokenKind_ != TokenKind::DT) { // 13.3.2p6
      ReportError("String edit descriptor in READ format expression");
    } else if (token_.kind() != TokenKind::String) {
      ReportError("Unterminated string");
    }
    break;
  default:
    if (cursor_ >= end_ && !unterminatedFormatError_) {
      suppressMessageCascade_ = false;
      ReportError("Unterminated format expression");
      unterminatedFormatError_ = true;
    }
    token_.set_kind(TokenKind::None);
    break;
  }

  SetLength();
}

template <typename CHAR> void FormatValidator<CHAR>::check_r(bool allowed) {
  if (!allowed && knrValue_ >= 0) {
    ReportError("Repeat specifier before '%s' edit descriptor", knrToken_);
  } else if (knrValue_ == 0) {
    ReportError("'%s' edit descriptor repeat specifier must be positive",
        knrToken_); // C1304
  }
}

// Return the predicate "w value is present" to control further processing.
template <typename CHAR> bool FormatValidator<CHAR>::check_w() {
  if (token_.kind() == TokenKind::UnsignedInteger) {
    wValue_ = integerValue_;
    if (wValue_ == 0 &&
        (*argString_ == 'A' || *argString_ == 'L' ||
            stmt_ == IoStmtKind::Read)) { // C1306, 13.7.2.1p6
      ReportError("'%s' edit descriptor 'w' value must be positive");
    }
    NextToken();
    return true;
  }
  if (*argString_ != 'A') {
    ReportWarning("Expected '%s' edit descriptor 'w' value"); // C1306
  }
  return false;
}

template <typename CHAR> void FormatValidator<CHAR>::check_m() {
  if (token_.kind() != TokenKind::Point) {
    return;
  }
  NextToken();
  if (token_.kind() != TokenKind::UnsignedInteger) {
    ReportError("Expected '%s' edit descriptor 'm' value after '.'");
    return;
  }
  if ((stmt_ == IoStmtKind::Print || stmt_ == IoStmtKind::Write) &&
      wValue_ > 0 && integerValue_ > wValue_) { // 13.7.2.2p5, 13.7.2.4p6
    ReportError("'%s' edit descriptor 'm' value is greater than 'w' value");
  }
  NextToken();
}

// Return the predicate "d value is present" to control further processing.
template <typename CHAR> bool FormatValidator<CHAR>::check_d() {
  if (token_.kind() != TokenKind::Point) {
    ReportError("Expected '%s' edit descriptor '.d' value");
    return false;
  }
  NextToken();
  if (token_.kind() != TokenKind::UnsignedInteger) {
    ReportError("Expected '%s' edit descriptor 'd' value after '.'");
    return false;
  }
  NextToken();
  return true;
}

template <typename CHAR> void FormatValidator<CHAR>::check_e() {
  if (token_.kind() != TokenKind::E) {
    return;
  }
  NextToken();
  if (token_.kind() != TokenKind::UnsignedInteger) {
    ReportError("Expected '%s' edit descriptor 'e' value after 'E'");
    return;
  }
  NextToken();
}

template <typename CHAR> bool FormatValidator<CHAR>::Check() {
  if (!*format_) {
    ReportError("Empty format expression");
    return formatHasErrors_;
  }
  NextToken();
  if (token_.kind() != TokenKind::LParen) {
    ReportError("Format expression must have an initial '('");
    return formatHasErrors_;
  }
  NextToken();

  int nestLevel{0}; // Outer level ()s are at level 0.
  Token starToken{}; // unlimited format token
  bool hasDataEditDesc{false};

  // Subject to error recovery exceptions, a loop iteration processes one
  // edit descriptor or does list management.  The loop terminates when
  //  - a level-0 right paren is processed (format may be valid)
  //  - the end of an incomplete format is reached
  //  - the error reporter requests termination (error threshold reached)
  while (!reporterExit_) {
    Token signToken{};
    knrValue_ = -1; // -1 ==> not present
    wValue_ = -1;
    bool commaRequired{true};

    if (token_.kind() == TokenKind::Sign) {
      signToken = token_;
      NextToken();
    }
    if (token_.kind() == TokenKind::UnsignedInteger) {
      knrToken_ = token_;
      knrValue_ = integerValue_;
      NextToken();
    }
    if (signToken.IsSet() && (knrValue_ < 0 || token_.kind() != TokenKind::P)) {
      argString_[0] = format_[signToken.offset()];
      argString_[1] = 0;
      ReportError("Unexpected '%s' in format expression", signToken);
    }
    // Default message argument.
    // Alphabetic edit descriptor names are one or two characters in length.
    argString_[0] = toupper(format_[token_.offset()]);
    argString_[1] = token_.length() > 1 ? toupper(*cursor_) : 0;
    // Process one format edit descriptor or do format list management.
    switch (token_.kind()) {
    case TokenKind::A:
      // R1307 data-edit-desc -> A [w]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      check_w();
      break;
    case TokenKind::B:
    case TokenKind::I:
    case TokenKind::O:
    case TokenKind::Z:
      // R1307 data-edit-desc -> B w [. m] | I w [. m] | O w [. m] | Z w [. m]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w()) {
        check_m();
      }
      break;
    case TokenKind::D:
    case TokenKind::F:
      // R1307 data-edit-desc -> D w . d | F w . d
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w()) {
        check_d();
      }
      break;
    case TokenKind::E:
    case TokenKind::EN:
    case TokenKind::ES:
    case TokenKind::EX:
      // R1307 data-edit-desc ->
      //   E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w() && check_d()) {
        check_e();
      }
      break;
    case TokenKind::G:
      // R1307 data-edit-desc -> G w [. d [E e]]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (check_w()) {
        if (wValue_ > 0) {
          if (check_d()) { // C1307
            check_e();
          }
        } else if (token_.kind() == TokenKind::Point && check_d() &&
            token_.kind() == TokenKind::E) { // C1308
          ReportError("A 'G0' edit descriptor must not have an 'e' value");
          NextToken();
          if (token_.kind() == TokenKind::UnsignedInteger) {
            NextToken();
          }
        }
      }
      break;
    case TokenKind::L:
      // R1307 data-edit-desc -> L w
      hasDataEditDesc = true;
      check_r();
      NextToken();
      check_w();
      break;
    case TokenKind::DT:
      // R1307 data-edit-desc -> DT [char-literal-constant] [( v-list )]
      hasDataEditDesc = true;
      check_r();
      NextToken();
      if (token_.kind() == TokenKind::String) {
        NextToken();
      }
      if (token_.kind() == TokenKind::LParen) {
        do {
          NextToken();
          if (token_.kind() == TokenKind::Sign) {
            NextToken();
          }
          if (token_.kind() != TokenKind::UnsignedInteger) {
            ReportError(
                "Expected integer constant in 'DT' edit descriptor v-list");
            break;
          }
          NextToken();
        } while (token_.kind() == TokenKind::Comma);
        if (token_.kind() != TokenKind::RParen) {
          ReportError("Expected ',' or ')' in 'DT' edit descriptor v-list");
          while (cursor_ < end_ && token_.kind() != TokenKind::RParen) {
            NextToken();
          }
        }
        NextToken();
      }
      break;
    case TokenKind::String:
      // R1304 data-edit-desc -> char-string-edit-desc
      if (knrValue_ >= 0) {
        ReportError("Repeat specifier before character string edit descriptor",
            knrToken_);
      }
      NextToken();
      break;
    case TokenKind::BN:
    case TokenKind::BZ:
    case TokenKind::DC:
    case TokenKind::DP:
    case TokenKind::RC:
    case TokenKind::RD:
    case TokenKind::RN:
    case TokenKind::RP:
    case TokenKind::RU:
    case TokenKind::RZ:
    case TokenKind::S:
    case TokenKind::SP:
    case TokenKind::SS:
      // R1317 sign-edit-desc -> SS | SP | S
      // R1318 blank-interp-edit-desc -> BN | BZ
      // R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
      // R1320 decimal-edit-desc -> DC | DP
      check_r(false);
      NextToken();
      break;
    case TokenKind::P: {
      // R1313 control-edit-desc -> k P
      if (knrValue_ < 0) {
        ReportError("'P' edit descriptor must have a scale factor");
      }
      // Diagnosing C1302 may require multiple token lookahead.
      // Save current cursor position to enable backup.
      const CHAR *saveCursor{cursor_};
      NextToken();
      if (token_.kind() == TokenKind::UnsignedInteger) {
        NextToken();
      }
      switch (token_.kind()) {
      case TokenKind::D:
      case TokenKind::E:
      case TokenKind::EN:
      case TokenKind::ES:
      case TokenKind::EX:
      case TokenKind::F:
      case TokenKind::G:
        commaRequired = false;
        break;
      default:;
      }
      cursor_ = saveCursor;
      NextToken();
      break;
    }
    case TokenKind::T:
    case TokenKind::TL:
    case TokenKind::TR:
      // R1315 position-edit-desc -> T n | TL n | TR n
      check_r(false);
      NextToken();
      if (integerValue_ <= 0) { // C1311
        ReportError("'%s' edit descriptor must have a positive position value");
      }
      NextToken();
      break;
    case TokenKind::X:
      // R1315 position-edit-desc -> n X
      if (knrValue_ == 0) { // C1311
        ReportError("'X' edit descriptor must have a positive position value",
            knrToken_);
      } else if (knrValue_ < 0) {
        ReportWarning(
            "'X' edit descriptor must have a positive position value");
      }
      NextToken();
      break;
    case TokenKind::Colon:
      // R1313 control-edit-desc -> :
      check_r(false);
      commaRequired = false;
      NextToken();
      break;
    case TokenKind::Slash:
      // R1313 control-edit-desc -> [r] /
      commaRequired = false;
      NextToken();
      break;
    case TokenKind::Backslash:
      check_r(false);
      ReportWarning("Non-standard '\\' edit descriptor");
      NextToken();
      break;
    case TokenKind::Dollar:
      check_r(false);
      ReportWarning("Non-standard '$' edit descriptor");
      NextToken();
      break;
    case TokenKind::Star:
      // NextToken assigns a token kind of Star only if * is followed by (.
      // So the next token is guaranteed to be LParen.
      if (nestLevel > 0) {
        ReportError("Nested unlimited format item list");
      }
      starToken = token_;
      if (knrValue_ >= 0) {
        ReportError(
            "Repeat specifier before unlimited format item list", knrToken_);
      }
      hasDataEditDesc = false;
      NextToken();
      [[fallthrough]];
    case TokenKind::LParen:
      if (knrValue_ == 0) {
        ReportError("List repeat specifier must be positive", knrToken_);
      }
      if (++nestLevel > maxNesting_) {
        maxNesting_ = nestLevel;
      }
      break;
    case TokenKind::RParen:
      if (knrValue_ >= 0) {
        ReportError("Unexpected integer constant", knrToken_);
      }
      do {
        if (nestLevel == 0) {
          // Any characters after level-0 ) are ignored.
          return formatHasErrors_; // normal exit (may have messages)
        }
        if (nestLevel == 1 && starToken.IsSet() && !hasDataEditDesc) {
          SetLength(starToken);
          ReportError( // C1303
              "Unlimited format item list must contain a data edit descriptor",
              starToken);
        }
        --nestLevel;
        NextToken();
      } while (token_.kind() == TokenKind::RParen);
      if (nestLevel == 0 && starToken.IsSet()) {
        ReportError("Character in format after unlimited format item list");
      }
      break;
    case TokenKind::Comma:
      if (knrValue_ >= 0) {
        ReportError("Unexpected integer constant", knrToken_);
      }
      if (suppressMessageCascade_ || reporterExit_) {
        break;
      }
      [[fallthrough]];
    default:
      ReportError("Unexpected '%s' in format expression");
      NextToken();
    }

    // Process comma separator and exit an incomplete format.
    switch (token_.kind()) {
    case TokenKind::Colon: // Comma not required; token not yet processed.
    case TokenKind::Slash: // Comma not required; token not yet processed.
    case TokenKind::RParen: // Comma not allowed; token not yet processed.
      suppressMessageCascade_ = false;
      break;
    case TokenKind::LParen: // Comma not allowed; token already processed.
    case TokenKind::Comma: // Normal comma case; move past token.
      suppressMessageCascade_ = false;
      NextToken();
      break;
    case TokenKind::Sign: // Error; main switch has a better message.
    case TokenKind::None: // Error; token not yet processed.
      if (cursor_ >= end_) {
        return formatHasErrors_; // incomplete format error exit
      }
      break;
    default:
      // Possible first token of the next format item; token not yet processed.
      if (commaRequired) {
        const char *s{"Expected ',' or ')' in format expression"}; // C1302
        if (previousTokenKind_ == TokenKind::UnsignedInteger &&
            itemsWithLeadingInts_.test(token_.kind())) {
          ReportError(s);
        } else {
          ReportWarning(s);
        }
      }
    }
  }

  return formatHasErrors_; // error reporter (message threshold) exit
}

} // namespace Fortran::common
#endif // FORTRAN_COMMON_FORMAT_H_
