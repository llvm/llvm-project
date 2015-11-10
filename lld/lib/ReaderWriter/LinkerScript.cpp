//===- ReaderWriter/LinkerScript.cpp ----------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Linker script parser.
///
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/LinkerScript.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace script {
void Token::dump(raw_ostream &os) const {
  switch (_kind) {
#define CASE(name)                                                             \
  case Token::name:                                                            \
    os << #name ": ";                                                          \
    break;
    CASE(unknown)
    CASE(eof)
    CASE(exclaim)
    CASE(exclaimequal)
    CASE(amp)
    CASE(ampequal)
    CASE(l_paren)
    CASE(r_paren)
    CASE(star)
    CASE(starequal)
    CASE(plus)
    CASE(plusequal)
    CASE(comma)
    CASE(minus)
    CASE(minusequal)
    CASE(slash)
    CASE(slashequal)
    CASE(number)
    CASE(colon)
    CASE(semicolon)
    CASE(less)
    CASE(lessequal)
    CASE(lessless)
    CASE(lesslessequal)
    CASE(equal)
    CASE(equalequal)
    CASE(greater)
    CASE(greaterequal)
    CASE(greatergreater)
    CASE(greatergreaterequal)
    CASE(question)
    CASE(identifier)
    CASE(libname)
    CASE(kw_align)
    CASE(kw_align_with_input)
    CASE(kw_as_needed)
    CASE(kw_at)
    CASE(kw_discard)
    CASE(kw_entry)
    CASE(kw_exclude_file)
    CASE(kw_extern)
    CASE(kw_filehdr)
    CASE(kw_fill)
    CASE(kw_flags)
    CASE(kw_group)
    CASE(kw_hidden)
    CASE(kw_input)
    CASE(kw_keep)
    CASE(kw_length)
    CASE(kw_memory)
    CASE(kw_origin)
    CASE(kw_phdrs)
    CASE(kw_provide)
    CASE(kw_provide_hidden)
    CASE(kw_only_if_ro)
    CASE(kw_only_if_rw)
    CASE(kw_output)
    CASE(kw_output_arch)
    CASE(kw_output_format)
    CASE(kw_overlay)
    CASE(kw_search_dir)
    CASE(kw_sections)
    CASE(kw_sort_by_alignment)
    CASE(kw_sort_by_init_priority)
    CASE(kw_sort_by_name)
    CASE(kw_sort_none)
    CASE(kw_subalign)
    CASE(l_brace)
    CASE(pipe)
    CASE(pipeequal)
    CASE(r_brace)
    CASE(tilde)
#undef CASE
  }
  os << _range << "\n";
}

static llvm::ErrorOr<uint64_t> parseDecimal(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res *= 10;
    if (c < '0' || c > '9')
      return llvm::ErrorOr<uint64_t>(make_error_code(llvm::errc::io_error));
    res += c - '0';
  }
  return res;
}

static llvm::ErrorOr<uint64_t> parseOctal(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res <<= 3;
    if (c < '0' || c > '7')
      return llvm::ErrorOr<uint64_t>(make_error_code(llvm::errc::io_error));
    res += c - '0';
  }
  return res;
}

static llvm::ErrorOr<uint64_t> parseBinary(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res <<= 1;
    if (c != '0' && c != '1')
      return llvm::ErrorOr<uint64_t>(make_error_code(llvm::errc::io_error));
    res += c - '0';
  }
  return res;
}

static llvm::ErrorOr<uint64_t> parseHex(StringRef str) {
  uint64_t res = 0;
  for (auto &c : str) {
    res <<= 4;
    if (c >= '0' && c <= '9')
      res += c - '0';
    else if (c >= 'a' && c <= 'f')
      res += c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      res += c - 'A' + 10;
    else
      return llvm::ErrorOr<uint64_t>(make_error_code(llvm::errc::io_error));
  }
  return res;
}

static bool parseHexToByteStream(StringRef str, std::string &buf) {
  unsigned char byte = 0;
  bool dumpByte = str.size() % 2;
  for (auto &c : str) {
    byte <<= 4;
    if (c >= '0' && c <= '9')
      byte += c - '0';
    else if (c >= 'a' && c <= 'f')
      byte += c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      byte += c - 'A' + 10;
    else
      return false;
    if (!dumpByte) {
      dumpByte = true;
      continue;
    }
    buf.push_back(byte);
    byte = 0;
    dumpByte = false;
  }
  return !dumpByte;
}

static void dumpByteStream(raw_ostream &os, StringRef stream) {
  os << "0x";
  for (auto &c : stream) {
    unsigned char firstNibble = c >> 4 & 0xF;
    if (firstNibble > 9)
      os << (char) ('A' + firstNibble - 10);
    else
      os << (char) ('0' + firstNibble);
    unsigned char secondNibble = c & 0xF;
    if (secondNibble > 9)
      os << (char) ('A' + secondNibble - 10);
    else
      os << (char) ('0' + secondNibble);
  }
}

static llvm::ErrorOr<uint64_t> parseNum(StringRef str) {
  unsigned multiplier = 1;
  enum NumKind { decimal, hex, octal, binary };
  NumKind kind = llvm::StringSwitch<NumKind>(str)
                     .StartsWith("0x", hex)
                     .StartsWith("0X", hex)
                     .StartsWith("0", octal)
                     .Default(decimal);

  // Parse scale
  if (str.endswith("K")) {
    multiplier = 1 << 10;
    str = str.drop_back();
  } else if (str.endswith("M")) {
    multiplier = 1 << 20;
    str = str.drop_back();
  }

  // Parse type
  if (str.endswith_lower("o")) {
    kind = octal;
    str = str.drop_back();
  } else if (str.endswith_lower("h")) {
    kind = hex;
    str = str.drop_back();
  } else if (str.endswith_lower("d")) {
    kind = decimal;
    str = str.drop_back();
  } else if (str.endswith_lower("b")) {
    kind = binary;
    str = str.drop_back();
  }

  llvm::ErrorOr<uint64_t> res(0);
  switch (kind) {
  case hex:
    if (str.startswith_lower("0x"))
      str = str.drop_front(2);
    res = parseHex(str);
    break;
  case octal:
    res = parseOctal(str);
    break;
  case decimal:
    res = parseDecimal(str);
    break;
  case binary:
    res = parseBinary(str);
    break;
  }
  if (res.getError())
    return res;

  *res = *res * multiplier;
  return res;
}

bool Lexer::canStartNumber(char c) const {
  return '0' <= c && c <= '9';
}

bool Lexer::canContinueNumber(char c) const {
  // [xX] = hex marker, [hHoO] = type suffix, [MK] = scale suffix.
  return strchr("0123456789ABCDEFabcdefxXhHoOMK", c);
}

bool Lexer::canStartName(char c) const {
  return strchr(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_.$/\\*", c);
}

bool Lexer::canContinueName(char c) const {
  return strchr("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "0123456789_.$/\\~=+[]*?-:", c);
}

/// Helper function to split a StringRef in two at the nth character.
/// The StringRef s is updated, while the function returns the n first
/// characters.
static StringRef drop(StringRef &s, int n) {
  StringRef res = s.substr(0, n);
  s = s.drop_front(n);
  return res;
}

void Lexer::lex(Token &tok) {
  skipWhitespace();
  if (_buffer.empty()) {
    tok = Token(_buffer, Token::eof);
    return;
  }
  switch (_buffer[0]) {
  case 0:
    tok = Token(drop(_buffer, 1), Token::eof);
    return;
  case '(':
    tok = Token(drop(_buffer, 1), Token::l_paren);
    return;
  case ')':
    tok = Token(drop(_buffer, 1), Token::r_paren);
    return;
  case '{':
    tok = Token(drop(_buffer, 1), Token::l_brace);
    return;
  case '}':
    tok = Token(drop(_buffer, 1), Token::r_brace);
    return;
  case '=':
    if (_buffer.startswith("==")) {
      tok = Token(drop(_buffer, 2), Token::equalequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::equal);
    return;
  case '!':
    if (_buffer.startswith("!=")) {
      tok = Token(drop(_buffer, 2), Token::exclaimequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::exclaim);
    return;
  case ',':
    tok = Token(drop(_buffer, 1), Token::comma);
    return;
  case ';':
    tok = Token(drop(_buffer, 1), Token::semicolon);
    return;
  case ':':
    tok = Token(drop(_buffer, 1), Token::colon);
    return;
  case '&':
    if (_buffer.startswith("&=")) {
      tok = Token(drop(_buffer, 2), Token::ampequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::amp);
    return;
  case '|':
    if (_buffer.startswith("|=")) {
      tok = Token(drop(_buffer, 2), Token::pipeequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::pipe);
    return;
  case '+':
    if (_buffer.startswith("+=")) {
      tok = Token(drop(_buffer, 2), Token::plusequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::plus);
    return;
  case '-': {
    if (_buffer.startswith("-=")) {
      tok = Token(drop(_buffer, 2), Token::minusequal);
      return;
    }
    if (!_buffer.startswith("-l")) {
      tok = Token(drop(_buffer, 1), Token::minus);
      return;
    }
    // -l<lib name>
    _buffer = _buffer.drop_front(2);
    StringRef::size_type start = 0;
    if (_buffer[start] == ':')
      ++start;
    if (!canStartName(_buffer[start]))
      // Create 'unknown' token.
      break;
    auto libNameEnd = std::find_if(_buffer.begin() + start + 1, _buffer.end(),
                                   [=](char c) { return !canContinueName(c); });
    StringRef::size_type libNameLen =
        std::distance(_buffer.begin(), libNameEnd);
    tok = Token(_buffer.substr(0, libNameLen), Token::libname);
    _buffer = _buffer.drop_front(libNameLen);
    return;
  }
  case '<':
    if (_buffer.startswith("<<=")) {
      tok = Token(drop(_buffer, 3), Token::lesslessequal);
      return;
    }
    if (_buffer.startswith("<<")) {
      tok = Token(drop(_buffer, 2), Token::lessless);
      return;
    }
    if (_buffer.startswith("<=")) {
      tok = Token(drop(_buffer, 2), Token::lessequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::less);
    return;
  case '>':
    if (_buffer.startswith(">>=")) {
      tok = Token(drop(_buffer, 3), Token::greatergreaterequal);
      return;
    }
    if (_buffer.startswith(">>")) {
      tok = Token(drop(_buffer, 2), Token::greatergreater);
      return;
    }
    if (_buffer.startswith(">=")) {
      tok = Token(drop(_buffer, 2), Token::greaterequal);
      return;
    }
    tok = Token(drop(_buffer, 1), Token::greater);
    return;
  case '~':
    tok = Token(drop(_buffer, 1), Token::tilde);
    return;
  case '\"': case '\'': {
    // Handle quoted strings. They are treated as identifiers for
    // simplicity.
    char c = _buffer[0];
    _buffer = _buffer.drop_front();
    auto quotedStringEnd = _buffer.find(c);
    if (quotedStringEnd == StringRef::npos || quotedStringEnd == 0)
      break;
    StringRef word = _buffer.substr(0, quotedStringEnd);
    tok = Token(word, Token::identifier);
    _buffer = _buffer.drop_front(quotedStringEnd + 1);
    return;
  }
  default:
    // Handle literal numbers
    if (canStartNumber(_buffer[0])) {
      auto endIter = std::find_if(_buffer.begin(), _buffer.end(), [=](char c) {
        return !canContinueNumber(c);
      });
      StringRef::size_type end = endIter == _buffer.end()
                                     ? StringRef::npos
                                     : std::distance(_buffer.begin(), endIter);
      if (end == StringRef::npos || end == 0)
        break;
      StringRef word = _buffer.substr(0, end);
      tok = Token(word, Token::number);
      _buffer = _buffer.drop_front(end);
      return;
    }
    // Handle slashes '/', which can be either an operator inside an expression
    // or the beginning of an identifier
    if (_buffer.startswith("/=")) {
      tok = Token(drop(_buffer, 2), Token::slashequal);
      return;
    }
    if (_buffer[0] == '/' && _buffer.size() > 1 &&
        !canContinueName(_buffer[1])) {
      tok = Token(drop(_buffer, 1), Token::slash);
      return;
    }
    // Handle stars '*'
    if (_buffer.startswith("*=")) {
      tok = Token(drop(_buffer, 2), Token::starequal);
      return;
    }
    if (_buffer[0] == '*' && _buffer.size() > 1 &&
        !canContinueName(_buffer[1])) {
      tok = Token(drop(_buffer, 1), Token::star);
      return;
    }
    // Handle questions '?'
    if (_buffer[0] == '?' && _buffer.size() > 1 &&
        !canContinueName(_buffer[1])) {
      tok = Token(drop(_buffer, 1), Token::question);
      return;
    }
    // keyword or identifier.
    if (!canStartName(_buffer[0]))
      break;
    auto endIter = std::find_if(_buffer.begin() + 1, _buffer.end(),
                                [=](char c) { return !canContinueName(c); });
    StringRef::size_type end = endIter == _buffer.end()
                                   ? StringRef::npos
                                   : std::distance(_buffer.begin(), endIter);
    if (end == StringRef::npos || end == 0)
      break;
    StringRef word = _buffer.substr(0, end);
    Token::Kind kind =
        llvm::StringSwitch<Token::Kind>(word)
            .Case("ALIGN", Token::kw_align)
            .Case("ALIGN_WITH_INPUT", Token::kw_align_with_input)
            .Case("AS_NEEDED", Token::kw_as_needed)
            .Case("AT", Token::kw_at)
            .Case("ENTRY", Token::kw_entry)
            .Case("EXCLUDE_FILE", Token::kw_exclude_file)
            .Case("EXTERN", Token::kw_extern)
            .Case("FILEHDR", Token::kw_filehdr)
            .Case("FILL", Token::kw_fill)
            .Case("FLAGS", Token::kw_flags)
            .Case("GROUP", Token::kw_group)
            .Case("HIDDEN", Token::kw_hidden)
            .Case("INPUT", Token::kw_input)
            .Case("KEEP", Token::kw_keep)
            .Case("LENGTH", Token::kw_length)
            .Case("l", Token::kw_length)
            .Case("len", Token::kw_length)
            .Case("MEMORY", Token::kw_memory)
            .Case("ONLY_IF_RO", Token::kw_only_if_ro)
            .Case("ONLY_IF_RW", Token::kw_only_if_rw)
            .Case("ORIGIN", Token::kw_origin)
            .Case("o", Token::kw_origin)
            .Case("org", Token::kw_origin)
            .Case("OUTPUT", Token::kw_output)
            .Case("OUTPUT_ARCH", Token::kw_output_arch)
            .Case("OUTPUT_FORMAT", Token::kw_output_format)
            .Case("OVERLAY", Token::kw_overlay)
            .Case("PHDRS", Token::kw_phdrs)
            .Case("PROVIDE", Token::kw_provide)
            .Case("PROVIDE_HIDDEN", Token::kw_provide_hidden)
            .Case("SEARCH_DIR", Token::kw_search_dir)
            .Case("SECTIONS", Token::kw_sections)
            .Case("SORT", Token::kw_sort_by_name)
            .Case("SORT_BY_ALIGNMENT", Token::kw_sort_by_alignment)
            .Case("SORT_BY_INIT_PRIORITY", Token::kw_sort_by_init_priority)
            .Case("SORT_BY_NAME", Token::kw_sort_by_name)
            .Case("SORT_NONE", Token::kw_sort_none)
            .Case("SUBALIGN", Token::kw_subalign)
            .Case("/DISCARD/", Token::kw_discard)
            .Default(Token::identifier);
    tok = Token(word, kind);
    _buffer = _buffer.drop_front(end);
    return;
  }
  tok = Token(drop(_buffer, 1), Token::unknown);
}

void Lexer::skipWhitespace() {
  while (true) {
    if (_buffer.empty())
      return;
    switch (_buffer[0]) {
    case ' ':
    case '\r':
    case '\n':
    case '\t':
      _buffer = _buffer.drop_front();
      break;
    // Potential comment.
    case '/':
      if (_buffer.size() <= 1 || _buffer[1] != '*')
        return;
      // Skip starting /*
      _buffer = _buffer.drop_front(2);
      // If the next char is also a /, it's not the end.
      if (!_buffer.empty() && _buffer[0] == '/')
        _buffer = _buffer.drop_front();

      // Scan for /'s. We're done if it is preceded by a *.
      while (true) {
        if (_buffer.empty())
          break;
        _buffer = _buffer.drop_front();
        if (_buffer.data()[-1] == '/' && _buffer.data()[-2] == '*')
          break;
      }
      break;
    default:
      return;
    }
  }
}

// Constant functions
void Constant::dump(raw_ostream &os) const { os << _num; }

ErrorOr<int64_t> Constant::evalExpr(const SymbolTableTy &symbolTable) const {
  return _num;
}

// Symbol functions
void Symbol::dump(raw_ostream &os) const { os << _name; }

ErrorOr<int64_t> Symbol::evalExpr(const SymbolTableTy &symbolTable) const {
  auto it = symbolTable.find(_name);
  if (it == symbolTable.end())
    return LinkerScriptReaderError::unknown_symbol_in_expr;
  return it->second;
}

// FunctionCall functions
void FunctionCall::dump(raw_ostream &os) const {
  os << _name << "(";
  for (unsigned i = 0, e = _args.size(); i != e; ++i) {
    if (i)
      os << ", ";
    _args[i]->dump(os);
  }
  os << ")";
}

ErrorOr<int64_t>
FunctionCall::evalExpr(const SymbolTableTy &symbolTable) const {
  return LinkerScriptReaderError::unrecognized_function_in_expr;
}

// Unary functions
void Unary::dump(raw_ostream &os) const {
  os << "(";
  if (_op == Unary::Minus)
    os << "-";
  else
    os << "~";
  _child->dump(os);
  os << ")";
}

ErrorOr<int64_t> Unary::evalExpr(const SymbolTableTy &symbolTable) const {
  auto child = _child->evalExpr(symbolTable);
  if (child.getError())
    return child.getError();

  int64_t childRes = *child;
  switch (_op) {
  case Unary::Minus:
    return -childRes;
  case Unary::Not:
    return ~childRes;
  }

  llvm_unreachable("");
}

// BinOp functions
void BinOp::dump(raw_ostream &os) const {
  os << "(";
  _lhs->dump(os);
  os << " ";
  switch (_op) {
  case Sum:
    os << "+";
    break;
  case Sub:
    os << "-";
    break;
  case Mul:
    os << "*";
    break;
  case Div:
    os << "/";
    break;
  case Shl:
    os << "<<";
    break;
  case Shr:
    os << ">>";
    break;
  case And:
    os << "&";
    break;
  case Or:
    os << "|";
    break;
  case CompareEqual:
    os << "==";
    break;
  case CompareDifferent:
    os << "!=";
    break;
  case CompareLess:
    os << "<";
    break;
  case CompareGreater:
    os << ">";
    break;
  case CompareLessEqual:
    os << "<=";
    break;
  case CompareGreaterEqual:
    os << ">=";
    break;
  }
  os << " ";
  _rhs->dump(os);
  os << ")";
}

ErrorOr<int64_t> BinOp::evalExpr(const SymbolTableTy &symbolTable) const {
  auto lhs = _lhs->evalExpr(symbolTable);
  if (lhs.getError())
    return lhs.getError();
  auto rhs = _rhs->evalExpr(symbolTable);
  if (rhs.getError())
    return rhs.getError();

  int64_t lhsRes = *lhs;
  int64_t rhsRes = *rhs;

  switch(_op) {
  case And:                 return lhsRes & rhsRes;
  case CompareDifferent:    return lhsRes != rhsRes;
  case CompareEqual:        return lhsRes == rhsRes;
  case CompareGreater:      return lhsRes > rhsRes;
  case CompareGreaterEqual: return lhsRes >= rhsRes;
  case CompareLess:         return lhsRes < rhsRes;
  case CompareLessEqual:    return lhsRes <= rhsRes;
  case Div:                 return lhsRes / rhsRes;
  case Mul:                 return lhsRes * rhsRes;
  case Or:                  return lhsRes | rhsRes;
  case Shl:                 return lhsRes << rhsRes;
  case Shr:                 return lhsRes >> rhsRes;
  case Sub:                 return lhsRes - rhsRes;
  case Sum:                 return lhsRes + rhsRes;
  }

  llvm_unreachable("");
}

// TernaryConditional functions
void TernaryConditional::dump(raw_ostream &os) const {
  _conditional->dump(os);
  os << " ? ";
  _trueExpr->dump(os);
  os << " : ";
  _falseExpr->dump(os);
}

ErrorOr<int64_t>
TernaryConditional::evalExpr(const SymbolTableTy &symbolTable) const {
  auto conditional = _conditional->evalExpr(symbolTable);
  if (conditional.getError())
    return conditional.getError();
  if (*conditional)
    return _trueExpr->evalExpr(symbolTable);
  return _falseExpr->evalExpr(symbolTable);
}

// SymbolAssignment functions
void SymbolAssignment::dump(raw_ostream &os) const {
  int numParen = 0;

  if (_assignmentVisibility != Default) {
    switch (_assignmentVisibility) {
    case Hidden:
      os << "HIDDEN(";
      break;
    case Provide:
      os << "PROVIDE(";
      break;
    case ProvideHidden:
      os << "PROVIDE_HIDDEN(";
      break;
    default:
      llvm_unreachable("Unknown visibility");
    }
    ++numParen;
  }

  os << _symbol << " ";
  switch (_assignmentKind) {
  case Simple:
    os << "=";
    break;
  case Sum:
    os << "+=";
    break;
  case Sub:
    os << "-=";
    break;
  case Mul:
    os << "*=";
    break;
  case Div:
    os << "/=";
    break;
  case Shl:
    os << "<<=";
    break;
  case Shr:
    os << ">>=";
    break;
  case And:
    os << "&=";
    break;
  case Or:
    os << "|=";
    break;
  }

  os << " ";
  _expression->dump(os);
  if (numParen)
    os << ")";
  os << ";";
}

static int dumpSortDirectives(raw_ostream &os, WildcardSortMode sortMode) {
  switch (sortMode) {
  case WildcardSortMode::NA:
    return 0;
  case WildcardSortMode::ByName:
    os << "SORT_BY_NAME(";
    return 1;
  case WildcardSortMode::ByAlignment:
    os << "SORT_BY_ALIGNMENT(";
    return 1;
  case WildcardSortMode::ByInitPriority:
    os << "SORT_BY_INIT_PRIORITY(";
    return 1;
  case WildcardSortMode::ByNameAndAlignment:
    os << "SORT_BY_NAME(SORT_BY_ALIGNMENT(";
    return 2;
  case WildcardSortMode::ByAlignmentAndName:
    os << "SORT_BY_ALIGNMENT(SORT_BY_NAME(";
    return 2;
  case WildcardSortMode::None:
    os << "SORT_NONE(";
    return 1;
  }
  return 0;
}

// InputSectionName functions
void InputSectionName::dump(raw_ostream &os) const {
  os << _name;
}

// InputSectionSortedGroup functions
static void dumpInputSections(raw_ostream &os,
                              llvm::ArrayRef<const InputSection *> secs) {
  bool excludeFile = false;
  bool first = true;

  for (auto &secName : secs) {
    if (!first)
      os << " ";
    first = false;
    // Coalesce multiple input sections marked with EXCLUDE_FILE in the same
    // EXCLUDE_FILE() group
    if (auto inputSec = dyn_cast<InputSectionName>(secName)) {
      if (!excludeFile && inputSec->hasExcludeFile()) {
        excludeFile = true;
        os << "EXCLUDE_FILE(";
      } else if (excludeFile && !inputSec->hasExcludeFile()) {
        excludeFile = false;
        os << ") ";
      }
    }
    secName->dump(os);
  }

  if (excludeFile)
    os << ")";
}

void InputSectionSortedGroup::dump(raw_ostream &os) const {
  int numParen = dumpSortDirectives(os, _sortMode);
  dumpInputSections(os, _sections);
  for (int i = 0; i < numParen; ++i)
    os << ")";
}

// InputSectionsCmd functions
void InputSectionsCmd::dump(raw_ostream &os) const {
  if (_keep)
    os << "KEEP(";

  int numParen = dumpSortDirectives(os, _fileSortMode);
  os << _memberName;
  for (int i = 0; i < numParen; ++i)
    os << ")";

  if (_archiveName.size() > 0) {
    os << ":";
    numParen = dumpSortDirectives(os, _archiveSortMode);
    os << _archiveName;
    for (int i = 0; i < numParen; ++i)
      os << ")";
  }

  if (_sections.size() > 0) {
    os << "(";
    dumpInputSections(os, _sections);
    os << ")";
  }

  if (_keep)
    os << ")";
}

void FillCmd::dump(raw_ostream &os) const {
  os << "FILL(";
  dumpByteStream(os, StringRef((const char *)_bytes.begin(), _bytes.size()));
  os << ")";
}

// OutputSectionDescription functions
void OutputSectionDescription::dump(raw_ostream &os) const {
  if (_discard)
    os << "/DISCARD/";
  else
    os << _sectionName;

  if (_address) {
    os << " ";
    _address->dump(os);
  }
  os << " :\n";

  if (_at) {
    os << "  AT(";
    _at->dump(os);
    os << ")\n";
  }

  if (_align) {
    os << "  ALIGN(";
    _align->dump(os);
    os << ")\n";
  } else if (_alignWithInput) {
    os << " ALIGN_WITH_INPUT\n";
  }

  if (_subAlign) {
    os << "  SUBALIGN(";
    _subAlign->dump(os);
    os << ")\n";
  }

  switch (_constraint) {
  case C_None:
    break;
  case C_OnlyIfRO:
    os << "ONLY_IF_RO";
    break;
  case C_OnlyIfRW:
    os << "ONLY_IF_RW";
    break;
  }

  os << "  {\n";
  for (auto &command : _outputSectionCommands) {
    os << "    ";
    command->dump(os);
    os << "\n";
  }
  os << "  }";

  for (auto && phdr : _phdrs)
    os << " : " << phdr;

  if (_fillStream.size() > 0) {
    os << " =";
    dumpByteStream(os, _fillStream);
  } else if (_fillExpr) {
    os << " =";
    _fillExpr->dump(os);
  }
}

// Special header that discards output sections assigned to it.
static const PHDR PHDR_NONE("NONE", 0, false, false, nullptr, 0);

bool PHDR::isNone() const {
  return this == &PHDR_NONE;
}

void PHDR::dump(raw_ostream &os) const {
  os << _name << " " << _type;
  if (_includeFileHdr)
    os << " FILEHDR";
  if (_includePHDRs)
    os << " PHDRS";
  if (_at) {
    os << " AT (";
    _at->dump(os);
    os << ")";
  }
  if (_flags)
    os << " FLAGS (" << _flags << ")";
  os << ";\n";
}

void PHDRS::dump(raw_ostream &os) const {
  os << "PHDRS\n{\n";
  for (auto &&phdr : _phdrs) {
    phdr->dump(os);
  }
  os << "}\n";
}

// Sections functions
void Sections::dump(raw_ostream &os) const {
  os << "SECTIONS\n{\n";
  for (auto &command : _sectionsCommands) {
    command->dump(os);
    os << "\n";
  }
  os << "}\n";
}

// Memory functions
void MemoryBlock::dump(raw_ostream &os) const {
    os << _name;

    if (!_attr.empty())
      os << " (" << _attr << ")";

    os << " : ";

    os << "ORIGIN = ";
    _origin->dump(os);
    os << ", ";

    os << "LENGTH = ";
    _length->dump(os);
}

void Memory::dump(raw_ostream &os) const {
  os << "MEMORY\n{\n";
  for (auto &block : _blocks) {
    block->dump(os);
    os << "\n";
  }
  os << "}\n";
}

// Extern functions
void Extern::dump(raw_ostream &os) const {
  os << "EXTERN(";
  for (unsigned i = 0, e = _symbols.size(); i != e; ++i) {
    if (i)
      os << " ";
    os << _symbols[i];
  }
  os << ")\n";
}

// Parser functions
std::error_code Parser::parse() {
  // Get the first token.
  _lex.lex(_tok);
  // Parse top level commands.
  while (true) {
    switch (_tok._kind) {
    case Token::eof:
      return std::error_code();
    case Token::semicolon:
      consumeToken();
      break;
    case Token::kw_output: {
      auto output = parseOutput();
      if (!output)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(output);
      break;
    }
    case Token::kw_output_format: {
      auto outputFormat = parseOutputFormat();
      if (!outputFormat)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(outputFormat);
      break;
    }
    case Token::kw_output_arch: {
      auto outputArch = parseOutputArch();
      if (!outputArch)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(outputArch);
      break;
    }
    case Token::kw_input: {
      Input *input = parsePathList<Input>();
      if (!input)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(input);
      break;
    }
    case Token::kw_group: {
      Group *group = parsePathList<Group>();
      if (!group)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(group);
      break;
    }
    case Token::kw_as_needed:
      // Not allowed at top level.
      error(_tok, "AS_NEEDED not allowed at top level.");
      return LinkerScriptReaderError::parse_error;
    case Token::kw_entry: {
      Entry *entry = parseEntry();
      if (!entry)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(entry);
      break;
    }
    case Token::kw_phdrs: {
      PHDRS *phdrs = parsePHDRS();
      if (!phdrs)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(phdrs);
      break;
    }
    case Token::kw_search_dir: {
      SearchDir *searchDir = parseSearchDir();
      if (!searchDir)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(searchDir);
      break;
    }
    case Token::kw_sections: {
      Sections *sections = parseSections();
      if (!sections)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(sections);
      break;
    }
    case Token::identifier:
    case Token::kw_hidden:
    case Token::kw_provide:
    case Token::kw_provide_hidden: {
      const Command *cmd = parseSymbolAssignment();
      if (!cmd)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(cmd);
      break;
    }
    case Token::kw_memory: {
      const Command *cmd = parseMemory();
      if (!cmd)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(cmd);
      break;
    }
    case Token::kw_extern: {
      const Command *cmd = parseExtern();
      if (!cmd)
        return LinkerScriptReaderError::parse_error;
      _script._commands.push_back(cmd);
      break;
    }
    default:
      // Unexpected.
      error(_tok, "expected linker script command");
      return LinkerScriptReaderError::parse_error;
    }
  }
  return LinkerScriptReaderError::parse_error;
}

const Expression *Parser::parseFunctionCall() {
  assert((_tok._kind == Token::identifier || _tok._kind == Token::kw_align) &&
         "expected function call first tokens");
  SmallVector<const Expression *, 8> params;
  StringRef name = _tok._range;

  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  if (_tok._kind == Token::r_paren) {
    consumeToken();
    return new (_alloc) FunctionCall(*this, _tok._range, params);
  }

  if (const Expression *firstParam = parseExpression())
    params.push_back(firstParam);
  else
    return nullptr;

  while (_tok._kind == Token::comma) {
    consumeToken();
    if (const Expression *param = parseExpression())
      params.push_back(param);
    else
      return nullptr;
  }

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;
  return new (_alloc) FunctionCall(*this, name, params);
}

bool Parser::expectExprOperand() {
  if (!(_tok._kind == Token::identifier || _tok._kind == Token::number ||
        _tok._kind == Token::kw_align || _tok._kind == Token::l_paren ||
        _tok._kind == Token::minus || _tok._kind == Token::tilde)) {
    error(_tok, "expected symbol, number, minus, tilde or left parenthesis.");
    return false;
  }
  return true;
}

const Expression *Parser::parseExprOperand() {
  if (!expectExprOperand())
    return nullptr;

  switch (_tok._kind) {
  case Token::identifier: {
    if (peek()._kind== Token::l_paren)
      return parseFunctionCall();
    auto *sym = new (_alloc) Symbol(*this, _tok._range);
    consumeToken();
    return sym;
  }
  case Token::kw_align:
    return parseFunctionCall();
  case Token::minus:
    consumeToken();
    return new (_alloc) Unary(*this, Unary::Minus, parseExprOperand());
  case Token::tilde:
    consumeToken();
    return new (_alloc) Unary(*this, Unary::Not, parseExprOperand());
  case Token::number: {
    auto val = parseNum(_tok._range);
    if (val.getError()) {
      error(_tok, "Unrecognized number constant");
      return nullptr;
    }
    auto *c = new (_alloc) Constant(*this, *val);
    consumeToken();
    return c;
  }
  case Token::l_paren: {
    consumeToken();
    const Expression *expr = parseExpression();
    if (!expectAndConsume(Token::r_paren, "expected )"))
      return nullptr;
    return expr;
  }
  default:
    llvm_unreachable("Unknown token");
  }
}

static bool TokenToBinOp(const Token &tok, BinOp::Operation &op,
                         unsigned &precedence) {
  switch (tok._kind) {
  case Token::star:
    op = BinOp::Mul;
    precedence = 3;
    return true;
  case Token::slash:
    op = BinOp::Div;
    precedence = 3;
    return true;
  case Token::plus:
    op = BinOp::Sum;
    precedence = 4;
    return true;
  case Token::minus:
    op = BinOp::Sub;
    precedence = 4;
    return true;
  case Token::lessless:
    op = BinOp::Shl;
    precedence = 5;
    return true;
  case Token::greatergreater:
    op = BinOp::Shr;
    precedence = 5;
    return true;
  case Token::less:
    op = BinOp::CompareLess;
    precedence = 6;
    return true;
  case Token::greater:
    op = BinOp::CompareGreater;
    precedence = 6;
    return true;
  case Token::lessequal:
    op = BinOp::CompareLessEqual;
    precedence = 6;
    return true;
  case Token::greaterequal:
    op = BinOp::CompareGreaterEqual;
    precedence = 6;
    return true;
  case Token::equalequal:
    op = BinOp::CompareEqual;
    precedence = 7;
    return true;
  case Token::exclaimequal:
    op = BinOp::CompareDifferent;
    precedence = 7;
    return true;
  case Token::amp:
    op = BinOp::And;
    precedence = 8;
    return true;
  case Token::pipe:
    op = BinOp::Or;
    precedence = 10;
    return true;
  default:
    break;
  }
  return false;
}

static bool isExpressionOperator(Token tok) {
  switch (tok._kind) {
  case Token::star:
  case Token::slash:
  case Token::plus:
  case Token::minus:
  case Token::lessless:
  case Token::greatergreater:
  case Token::less:
  case Token::greater:
  case Token::lessequal:
  case Token::greaterequal:
  case Token::equalequal:
  case Token::exclaimequal:
  case Token::amp:
  case Token::pipe:
  case Token::question:
    return true;
  default:
    return false;
  }
}

const Expression *Parser::parseExpression(unsigned precedence) {
  assert(precedence <= 13 && "Invalid precedence value");
  if (!expectExprOperand())
    return nullptr;

  const Expression *expr = parseExprOperand();
  if (!expr)
    return nullptr;

  BinOp::Operation op;
  unsigned binOpPrecedence = 0;
  if (TokenToBinOp(_tok, op, binOpPrecedence)) {
    if (precedence >= binOpPrecedence)
      return parseOperatorOperandLoop(expr, precedence);
    return expr;
  }

  // Non-binary operators
  if (_tok._kind == Token::question && precedence >= 13)
    return parseOperatorOperandLoop(expr, precedence);
  return expr;
}

const Expression *Parser::parseOperatorOperandLoop(const Expression *lhs,
                                                   unsigned highestPrecedence) {
  assert(highestPrecedence <= 13 && "Invalid precedence value");
  unsigned precedence = 0;
  const Expression *binOp = nullptr;

  while (1) {
    BinOp::Operation op;
    if (!TokenToBinOp(_tok, op, precedence)) {
      if (_tok._kind == Token::question && highestPrecedence >= 13)
        return parseTernaryCondOp(lhs);
      return binOp;
    }

    if (precedence > highestPrecedence)
      return binOp;

    consumeToken();
    const Expression *rhs = parseExpression(precedence - 1);
    if (!rhs)
      return nullptr;
    binOp = new (_alloc) BinOp(*this, lhs, op, rhs);
    lhs = binOp;
  }
}

const Expression *Parser::parseTernaryCondOp(const Expression *lhs) {
  assert(_tok._kind == Token::question && "Expected question mark");

  consumeToken();

  // The ternary conditional operator has right-to-left associativity.
  // To implement this, we allow our children to contain ternary conditional
  // operators themselves (precedence 13).
  const Expression *trueExpr = parseExpression(13);
  if (!trueExpr)
    return nullptr;

  if (!expectAndConsume(Token::colon, "expected :"))
    return nullptr;

  const Expression *falseExpr = parseExpression(13);
  if (!falseExpr)
    return nullptr;

  return new (_alloc) TernaryConditional(*this, lhs, trueExpr, falseExpr);
}

// Parse OUTPUT(ident)
Output *Parser::parseOutput() {
  assert(_tok._kind == Token::kw_output && "Expected OUTPUT");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  if (_tok._kind != Token::identifier) {
    error(_tok, "Expected identifier in OUTPUT.");
    return nullptr;
  }

  auto ret = new (_alloc) Output(*this, _tok._range);
  consumeToken();

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;

  return ret;
}

// Parse OUTPUT_FORMAT(ident)
OutputFormat *Parser::parseOutputFormat() {
  assert(_tok._kind == Token::kw_output_format && "Expected OUTPUT_FORMAT!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  if (_tok._kind != Token::identifier) {
    error(_tok, "Expected identifier in OUTPUT_FORMAT.");
    return nullptr;
  }

  SmallVector<StringRef, 8> formats;
  formats.push_back(_tok._range);

  consumeToken();

  do {
    if (isNextToken(Token::comma))
      consumeToken();
    else
      break;
    if (_tok._kind != Token::identifier) {
      error(_tok, "Expected identifier in OUTPUT_FORMAT.");
      return nullptr;
    }
    formats.push_back(_tok._range);
    consumeToken();
  } while (isNextToken(Token::comma));

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;

  return new (_alloc) OutputFormat(*this, formats);
}

// Parse OUTPUT_ARCH(ident)
OutputArch *Parser::parseOutputArch() {
  assert(_tok._kind == Token::kw_output_arch && "Expected OUTPUT_ARCH!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  if (_tok._kind != Token::identifier) {
    error(_tok, "Expected identifier in OUTPUT_ARCH.");
    return nullptr;
  }

  auto ret = new (_alloc) OutputArch(*this, _tok._range);
  consumeToken();

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;

  return ret;
}

// Parse file list for INPUT or GROUP
template<class T> T *Parser::parsePathList() {
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  SmallVector<Path, 8> paths;
  while (_tok._kind == Token::identifier || _tok._kind == Token::libname ||
         _tok._kind == Token::kw_as_needed) {
    switch (_tok._kind) {
    case Token::identifier:
      paths.push_back(Path(_tok._range));
      consumeToken();
      break;
    case Token::libname:
      paths.push_back(Path(_tok._range, false, true));
      consumeToken();
      break;
    case Token::kw_as_needed:
      if (!parseAsNeeded(paths))
        return nullptr;
      break;
    default:
      llvm_unreachable("Invalid token.");
    }
  }
  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;
  return new (_alloc) T(*this, paths);
}

// Parse AS_NEEDED(file ...)
bool Parser::parseAsNeeded(SmallVectorImpl<Path> &paths) {
  assert(_tok._kind == Token::kw_as_needed && "Expected AS_NEEDED!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return false;

  while (_tok._kind == Token::identifier || _tok._kind == Token::libname) {
    switch (_tok._kind) {
    case Token::identifier:
      paths.push_back(Path(_tok._range, true, false));
      consumeToken();
      break;
    case Token::libname:
      paths.push_back(Path(_tok._range, true, true));
      consumeToken();
      break;
    default:
      llvm_unreachable("Invalid token.");
    }
  }

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return false;
  return true;
}

// Parse ENTRY(ident)
Entry *Parser::parseEntry() {
  assert(_tok._kind == Token::kw_entry && "Expected ENTRY!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;
  if (_tok._kind != Token::identifier) {
    error(_tok, "expected identifier in ENTRY");
    return nullptr;
  }
  StringRef entryName(_tok._range);
  consumeToken();
  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;
  return new (_alloc) Entry(*this, entryName);
}

// Parse SEARCH_DIR(ident)
SearchDir *Parser::parseSearchDir() {
  assert(_tok._kind == Token::kw_search_dir && "Expected SEARCH_DIR!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;
  if (_tok._kind != Token::identifier) {
    error(_tok, "expected identifier in SEARCH_DIR");
    return nullptr;
  }
  StringRef searchPath(_tok._range);
  consumeToken();
  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;
  return new (_alloc) SearchDir(*this, searchPath);
}

const SymbolAssignment *Parser::parseSymbolAssignment() {
  assert((_tok._kind == Token::identifier || _tok._kind == Token::kw_hidden ||
          _tok._kind == Token::kw_provide ||
          _tok._kind == Token::kw_provide_hidden) &&
         "Expected identifier!");
  SymbolAssignment::AssignmentVisibility visibility = SymbolAssignment::Default;
  SymbolAssignment::AssignmentKind kind;
  int numParen = 0;

  switch (_tok._kind) {
  case Token::kw_hidden:
    visibility = SymbolAssignment::Hidden;
    ++numParen;
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return nullptr;
    break;
  case Token::kw_provide:
    visibility = SymbolAssignment::Provide;
    ++numParen;
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return nullptr;
    break;
  case Token::kw_provide_hidden:
    visibility = SymbolAssignment::ProvideHidden;
    ++numParen;
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return nullptr;
    break;
  default:
    break;
  }

  StringRef name = _tok._range;
  consumeToken();

  // Parse assignment operator (=, +=, -= etc.)
  switch (_tok._kind) {
  case Token::equal:
    kind = SymbolAssignment::Simple;
    break;
  case Token::plusequal:
    kind = SymbolAssignment::Sum;
    break;
  case Token::minusequal:
    kind = SymbolAssignment::Sub;
    break;
  case Token::starequal:
    kind = SymbolAssignment::Mul;
    break;
  case Token::slashequal:
    kind = SymbolAssignment::Div;
    break;
  case Token::ampequal:
    kind = SymbolAssignment::And;
    break;
  case Token::pipeequal:
    kind = SymbolAssignment::Or;
    break;
  case Token::lesslessequal:
    kind = SymbolAssignment::Shl;
    break;
  case Token::greatergreaterequal:
    kind = SymbolAssignment::Shr;
    break;
  default:
    error(_tok, "unexpected token");
    return nullptr;
  }

  consumeToken();

  const Expression *expr = nullptr;
  switch (_tok._kind) {
  case Token::number:
  case Token::kw_align:
  case Token::identifier:
  case Token::l_paren:
    expr = parseExpression();
    if (!expr)
      return nullptr;
    break;
  default:
    error(_tok, "unexpected token while parsing assignment value.");
    return nullptr;
  }

  for (int i = 0; i < numParen; ++i)
    if (!expectAndConsume(Token::r_paren, "expected )"))
      return nullptr;

  return new (_alloc) SymbolAssignment(*this, name, expr, kind, visibility);
}

llvm::ErrorOr<InputSectionsCmd::VectorTy> Parser::parseExcludeFile() {
  assert(_tok._kind == Token::kw_exclude_file && "Expected EXCLUDE_FILE!");
  InputSectionsCmd::VectorTy res;
  consumeToken();

  if (!expectAndConsume(Token::l_paren, "expected ("))
    return llvm::ErrorOr<InputSectionsCmd::VectorTy>(
        make_error_code(llvm::errc::io_error));

  while (_tok._kind == Token::identifier) {
    res.push_back(new (_alloc) InputSectionName(*this, _tok._range, true));
    consumeToken();
  }

  if (!expectAndConsume(Token::r_paren, "expected )"))
    return llvm::ErrorOr<InputSectionsCmd::VectorTy>(
        make_error_code(llvm::errc::io_error));
  return llvm::ErrorOr<InputSectionsCmd::VectorTy>(std::move(res));
}

int Parser::parseSortDirectives(WildcardSortMode &sortMode) {
  int numParsedDirectives = 0;
  sortMode = WildcardSortMode::NA;

  if (_tok._kind == Token::kw_sort_by_name) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return -1;
    ++numParsedDirectives;
    sortMode = WildcardSortMode::ByName;
  }

  if (_tok._kind == Token::kw_sort_by_init_priority) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return -1;
    ++numParsedDirectives;
    sortMode = WildcardSortMode::ByInitPriority;
  }

  if (_tok._kind == Token::kw_sort_by_alignment) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return -1;
    ++numParsedDirectives;
    if (sortMode != WildcardSortMode::ByName)
      sortMode = WildcardSortMode::ByAlignment;
    else
      sortMode = WildcardSortMode::ByNameAndAlignment;
  }

  if (numParsedDirectives < 2 && _tok._kind == Token::kw_sort_by_name) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return -1;
    ++numParsedDirectives;
    if (sortMode == WildcardSortMode::ByAlignment)
      sortMode = WildcardSortMode::ByAlignmentAndName;
  }

  if (numParsedDirectives < 2 && _tok._kind == Token::kw_sort_by_alignment) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return -1;
    ++numParsedDirectives;
  }

  if (numParsedDirectives == 0 && _tok._kind == Token::kw_sort_none) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return -1;
    ++numParsedDirectives;
    sortMode = WildcardSortMode::None;
  }

  return numParsedDirectives;
}

const InputSection *Parser::parseSortedInputSections() {
  assert((_tok._kind == Token::kw_sort_by_name ||
          _tok._kind == Token::kw_sort_by_alignment ||
          _tok._kind == Token::kw_sort_by_init_priority ||
          _tok._kind == Token::kw_sort_none) &&
         "Expected SORT directives!");

  WildcardSortMode sortMode = WildcardSortMode::NA;
  int numParen = parseSortDirectives(sortMode);
  if (numParen == -1)
    return nullptr;

  SmallVector<const InputSection *, 8> inputSections;

  while (_tok._kind == Token::identifier) {
    inputSections.push_back(new (_alloc)
                                InputSectionName(*this, _tok._range, false));
    consumeToken();
  }

  // Eat "numParen" rparens
  for (int i = 0, e = numParen; i != e; ++i)
    if (!expectAndConsume(Token::r_paren, "expected )"))
      return nullptr;

  return new (_alloc) InputSectionSortedGroup(*this, sortMode, inputSections);
}

const InputSectionsCmd *Parser::parseInputSectionsCmd() {
  assert((_tok._kind == Token::identifier || _tok._kind == Token::colon ||
          _tok._kind == Token::star || _tok._kind == Token::kw_keep ||
          _tok._kind == Token::kw_sort_by_name ||
          _tok._kind == Token::kw_sort_by_alignment ||
          _tok._kind == Token::kw_sort_by_init_priority ||
          _tok._kind == Token::kw_sort_none) &&
         "Expected input section first tokens!");
  int numParen = 1;
  bool keep = false;
  WildcardSortMode fileSortMode = WildcardSortMode::NA;
  WildcardSortMode archiveSortMode = WildcardSortMode::NA;
  StringRef memberName;
  StringRef archiveName;

  if (_tok._kind == Token::kw_keep) {
    consumeToken();
    if (!expectAndConsume(Token::l_paren, "expected ("))
      return nullptr;
    ++numParen;
    keep = true;
  }

  // Input name
  if (_tok._kind != Token::colon) {
    int numParen = parseSortDirectives(fileSortMode);
    if (numParen == -1)
      return nullptr;
    memberName = _tok._range;
    consumeToken();
    if (numParen) {
      while (numParen--)
        if (!expectAndConsume(Token::r_paren, "expected )"))
          return nullptr;
    }
  }
  if (_tok._kind == Token::colon) {
    consumeToken();
    if (_tok._kind == Token::identifier ||
        _tok._kind == Token::kw_sort_by_name ||
        _tok._kind == Token::kw_sort_by_alignment ||
        _tok._kind == Token::kw_sort_by_init_priority ||
        _tok._kind == Token::kw_sort_none) {
      int numParen = parseSortDirectives(archiveSortMode);
      if (numParen == -1)
        return nullptr;
      archiveName = _tok._range;
      consumeToken();
      for (int i = 0; i != numParen; ++i)
	if (!expectAndConsume(Token::r_paren, "expected )"))
	  return nullptr;
    }
  }

  SmallVector<const InputSection *, 8> inputSections;

  if (_tok._kind != Token::l_paren)
    return new (_alloc)
        InputSectionsCmd(*this, memberName, archiveName, keep, fileSortMode,
                         archiveSortMode, inputSections);
  consumeToken();

  while (_tok._kind == Token::identifier ||
         _tok._kind == Token::kw_exclude_file ||
         _tok._kind == Token::kw_sort_by_name ||
         _tok._kind == Token::kw_sort_by_alignment ||
         _tok._kind == Token::kw_sort_by_init_priority ||
         _tok._kind == Token::kw_sort_none) {
    switch (_tok._kind) {
    case Token::kw_exclude_file: {
      auto vec = parseExcludeFile();
      if (vec.getError())
        return nullptr;
      inputSections.insert(inputSections.end(), vec->begin(), vec->end());
      break;
    }
    case Token::star:
    case Token::identifier: {
      inputSections.push_back(new (_alloc)
                                  InputSectionName(*this, _tok._range, false));
      consumeToken();
      break;
    }
    case Token::kw_sort_by_name:
    case Token::kw_sort_by_alignment:
    case Token::kw_sort_by_init_priority:
    case Token::kw_sort_none: {
      const InputSection *group = parseSortedInputSections();
      if (!group)
        return nullptr;
      inputSections.push_back(group);
      break;
    }
    default:
      llvm_unreachable("Unknown token");
    }
  }

  for (int i = 0; i < numParen; ++i)
    if (!expectAndConsume(Token::r_paren, "expected )"))
      return nullptr;
  return new (_alloc)
      InputSectionsCmd(*this, memberName, archiveName, keep, fileSortMode,
                       archiveSortMode, inputSections);
}

const FillCmd *Parser::parseFillCmd() {
  assert(_tok._kind == Token::kw_fill && "Expected FILL!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  SmallVector<uint8_t, 8> storage;

  // If the expression is just a number, it's arbitrary length.
  if (_tok._kind == Token::number && peek()._kind == Token::r_paren) {
    if (_tok._range.size() > 2 && _tok._range.startswith("0x")) {
      StringRef num = _tok._range.substr(2);
      for (char c : num) {
        unsigned nibble = llvm::hexDigitValue(c);
        if (nibble == -1u)
          goto not_simple_hex;
        storage.push_back(nibble);
      }
      
      if (storage.size() % 2 != 0)
        storage.insert(storage.begin(), 0);

      // Collapse nibbles.
      for (std::size_t i = 0, e = storage.size() / 2; i != e; ++i)
        storage[i] = (storage[i * 2] << 4) + storage[(i * 2) + 1];

      storage.resize(storage.size() / 2);
    }
  }
not_simple_hex:

  const Expression *expr = parseExpression();
  if (!expr)
    return nullptr;
  if (!expectAndConsume(Token::r_paren, "expected )"))
    return nullptr;
  
  return new(getAllocator()) FillCmd(*this, storage);
}

const OutputSectionDescription *Parser::parseOutputSectionDescription() {
  assert((_tok._kind == Token::kw_discard || _tok._kind == Token::identifier) &&
         "Expected /DISCARD/ or identifier!");
  StringRef sectionName;
  const Expression *address = nullptr;
  const Expression *align = nullptr;
  const Expression *subAlign = nullptr;
  const Expression *at = nullptr;
  const Expression *fillExpr = nullptr;
  StringRef fillStream;
  bool alignWithInput = false;
  bool discard = false;
  OutputSectionDescription::Constraint constraint =
      OutputSectionDescription::C_None;
  SmallVector<const Command *, 8> outputSectionCommands;

  if (_tok._kind == Token::kw_discard)
    discard = true;
  else
    sectionName = _tok._range;
  consumeToken();

  if (_tok._kind == Token::number || _tok._kind == Token::identifier ||
      _tok._kind == Token::kw_align || _tok._kind == Token::l_paren) {
    address = parseExpression();
    if (!address)
      return nullptr;
  }

  if (!expectAndConsume(Token::colon, "expected :"))
    return nullptr;

  if (_tok._kind == Token::kw_at) {
    consumeToken();
    at = parseExpression();
    if (!at)
      return nullptr;
  }

  if (_tok._kind == Token::kw_align) {
    consumeToken();
    align = parseExpression();
    if (!align)
      return nullptr;
  }

  if (_tok._kind == Token::kw_align_with_input) {
    consumeToken();
    alignWithInput = true;
  }

  if (_tok._kind == Token::kw_subalign) {
    consumeToken();
    subAlign = parseExpression();
    if (!subAlign)
      return nullptr;
  }

  if (_tok._kind == Token::kw_only_if_ro) {
    consumeToken();
    constraint = OutputSectionDescription::C_OnlyIfRO;
  } else if (_tok._kind == Token::kw_only_if_rw) {
    consumeToken();
    constraint = OutputSectionDescription::C_OnlyIfRW;
  }

  if (!expectAndConsume(Token::l_brace, "expected {"))
    return nullptr;

  // Parse zero or more output-section-commands
  while (_tok._kind != Token::r_brace) {
    switch (_tok._kind) {
    case Token::semicolon:
      consumeToken();
      break;
    case Token::identifier:
      switch (peek()._kind) {
      case Token::equal:
      case Token::plusequal:
      case Token::minusequal:
      case Token::starequal:
      case Token::slashequal:
      case Token::ampequal:
      case Token::pipeequal:
      case Token::lesslessequal:
      case Token::greatergreaterequal:
        if (const Command *cmd = parseSymbolAssignment())
          outputSectionCommands.push_back(cmd);
        else
          return nullptr;
        break;
      default:
        if (const Command *cmd = parseInputSectionsCmd())
          outputSectionCommands.push_back(cmd);
        else
          return nullptr;
        break;
      }
      break;
    case Token::kw_fill:
      if (const Command *cmd = parseFillCmd())
        outputSectionCommands.push_back(cmd);
      else
        return nullptr;
      break;
    case Token::kw_keep:
    case Token::star:
    case Token::colon:
    case Token::kw_sort_by_name:
    case Token::kw_sort_by_alignment:
    case Token::kw_sort_by_init_priority:
    case Token::kw_sort_none:
      if (const Command *cmd = parseInputSectionsCmd())
        outputSectionCommands.push_back(cmd);
      else
        return nullptr;
      break;
    case Token::kw_hidden:
    case Token::kw_provide:
    case Token::kw_provide_hidden:
      if (const Command *cmd = parseSymbolAssignment())
        outputSectionCommands.push_back(cmd);
      else
        return nullptr;
      break;
    default:
      error(_tok, "expected symbol assignment or input file name.");
      return nullptr;
    }
  }

  if (!expectAndConsume(Token::r_brace, "expected }"))
    return nullptr;

  SmallVector<StringRef, 2> phdrs;
  while (_tok._kind == Token::colon) {
    consumeToken();
    if (_tok._kind != Token::identifier) {
      error(_tok, "expected program header name");
      return nullptr;
    }
    phdrs.push_back(_tok._range);
    consumeToken();
  }

  if (_tok._kind == Token::equal) {
    consumeToken();
    if (_tok._kind != Token::number || !_tok._range.startswith_lower("0x")) {
      fillExpr = parseExpression();
      if (!fillExpr)
        return nullptr;
    } else {
      std::string strBuf;
      if (isExpressionOperator(peek()) ||
          !parseHexToByteStream(_tok._range.drop_front(2), strBuf)) {
        fillExpr = parseExpression();
        if(!fillExpr)
          return nullptr;
      } else {
        char *rawBuf = (char *) _alloc.Allocate(strBuf.size(), 1);
        memcpy(rawBuf, strBuf.c_str(), strBuf.size());
        fillStream = StringRef(rawBuf, strBuf.size());
        consumeToken();
      }
    }
  }

  return new (_alloc) OutputSectionDescription(
      *this, sectionName, address, align, subAlign, at, fillExpr, fillStream,
      alignWithInput, discard, constraint, outputSectionCommands, phdrs);
}

const Overlay *Parser::parseOverlay() {
  assert(_tok._kind == Token::kw_overlay && "Expected OVERLAY!");
  error(_tok, "Overlay description is not yet supported.");
  return nullptr;
}

const PHDR *Parser::parsePHDR() {
  assert(_tok._kind == Token::identifier && "Expected identifier!");
  
  StringRef name = _tok._range;
  consumeToken();
    
  uint64_t type;

  switch (_tok._kind) {
  case Token::identifier:
  case Token::number:
  case Token::l_paren: {
    const Expression *expr = parseExpression();
    if (!expr)
      return nullptr;
    Expression::SymbolTableTy PHDRTypes;
#define PHDR_INSERT(x) PHDRTypes.insert(std::make_pair(#x, llvm::ELF::x))
    PHDR_INSERT(PT_NULL);
    PHDR_INSERT(PT_LOAD);
    PHDR_INSERT(PT_DYNAMIC);
    PHDR_INSERT(PT_INTERP);
    PHDR_INSERT(PT_NOTE);
    PHDR_INSERT(PT_SHLIB);
    PHDR_INSERT(PT_PHDR);
    PHDR_INSERT(PT_TLS);
    PHDR_INSERT(PT_LOOS);
    PHDR_INSERT(PT_GNU_EH_FRAME);
    PHDR_INSERT(PT_GNU_STACK);
    PHDR_INSERT(PT_GNU_RELRO);
    PHDR_INSERT(PT_SUNW_EH_FRAME);
    PHDR_INSERT(PT_SUNW_UNWIND);
    PHDR_INSERT(PT_HIOS);
    PHDR_INSERT(PT_LOPROC);
    PHDR_INSERT(PT_ARM_ARCHEXT);
    PHDR_INSERT(PT_ARM_EXIDX);
    PHDR_INSERT(PT_ARM_UNWIND);
    PHDR_INSERT(PT_MIPS_REGINFO);
    PHDR_INSERT(PT_MIPS_RTPROC);
    PHDR_INSERT(PT_MIPS_OPTIONS);
    PHDR_INSERT(PT_MIPS_ABIFLAGS);
    PHDR_INSERT(PT_HIPROC);
#undef PHDR_INSERT
    auto t = expr->evalExpr(PHDRTypes);
    if (t == LinkerScriptReaderError::unknown_symbol_in_expr) {
      error(_tok, "Unknown type");
      return nullptr;
    }
    if (!t)
      return nullptr;
    type = *t;
    break;
  }
  default:
    error(_tok, "expected identifier or expression");
    return nullptr;
  }

  uint64_t flags = 0;
  const Expression *flagsExpr = nullptr;
  bool includeFileHdr = false;
  bool includePHDRs = false;

  while (_tok._kind != Token::semicolon) {
    switch (_tok._kind) {
    case Token::kw_filehdr:
      if (includeFileHdr) {
        error(_tok, "Duplicate FILEHDR attribute");
        return nullptr;
      }
      includeFileHdr = true;
      consumeToken();
      break;
    case Token::kw_phdrs:
      if (includePHDRs) {
        error(_tok, "Duplicate PHDRS attribute");
        return nullptr;
      }
      includePHDRs = true;
      consumeToken();
      break;
    case Token::kw_flags: {
      if (flagsExpr) {
        error(_tok, "Duplicate FLAGS attribute");
        return nullptr;
      }
      consumeToken();
      if (!expectAndConsume(Token::l_paren, "Expected ("))
        return nullptr;
      flagsExpr = parseExpression();
      if (!flagsExpr)
        return nullptr;
      auto f = flagsExpr->evalExpr();
      if (!f)
        return nullptr;
      flags = *f;
      if (!expectAndConsume(Token::r_paren, "Expected )"))
        return nullptr;
    } break;
    default:
      error(_tok, "Unexpected token");
      return nullptr;
    }
  }
  
  if (!expectAndConsume(Token::semicolon, "Expected ;"))
    return nullptr;

  return new (getAllocator())
      PHDR(name, type, includeFileHdr, includePHDRs, nullptr, flags);
}

PHDRS *Parser::parsePHDRS() {
  assert(_tok._kind == Token::kw_phdrs && "Expected PHDRS!");
  consumeToken();
  if (!expectAndConsume(Token::l_brace, "expected {"))
    return nullptr;

  SmallVector<const PHDR *, 8> phdrs;

  while (true) {
    if (_tok._kind == Token::identifier) {
      const PHDR *phdr = parsePHDR();
      if (!phdr)
        return nullptr;
      phdrs.push_back(phdr);
    } else
      break;
  }

  if (!expectAndConsume(Token::r_brace, "expected }"))
    return nullptr;

  return new (getAllocator()) PHDRS(*this, phdrs);
}

Sections *Parser::parseSections() {
  assert(_tok._kind == Token::kw_sections && "Expected SECTIONS!");
  consumeToken();
  if (!expectAndConsume(Token::l_brace, "expected {"))
    return nullptr;
  SmallVector<const Command *, 8> sectionsCommands;

  bool unrecognizedToken = false;
  // Parse zero or more sections-commands
  while (!unrecognizedToken) {
    switch (_tok._kind) {
    case Token::semicolon:
      consumeToken();
      break;

    case Token::identifier:
      switch (peek()._kind) {
      case Token::equal:
      case Token::plusequal:
      case Token::minusequal:
      case Token::starequal:
      case Token::slashequal:
      case Token::ampequal:
      case Token::pipeequal:
      case Token::lesslessequal:
      case Token::greatergreaterequal:
        if (const Command *cmd = parseSymbolAssignment())
          sectionsCommands.push_back(cmd);
        else
          return nullptr;
        break;
      default:
        if (const Command *cmd = parseOutputSectionDescription())
          sectionsCommands.push_back(cmd);
        else
          return nullptr;
        break;
      }
      break;

    case Token::kw_discard:
    case Token::star:
      if (const Command *cmd = parseOutputSectionDescription())
        sectionsCommands.push_back(cmd);
      else
        return nullptr;
      break;

    case Token::kw_entry:
      if (const Command *cmd = parseEntry())
        sectionsCommands.push_back(cmd);
      else
        return nullptr;
      break;

    case Token::kw_hidden:
    case Token::kw_provide:
    case Token::kw_provide_hidden:
      if (const Command *cmd = parseSymbolAssignment())
        sectionsCommands.push_back(cmd);
      else
        return nullptr;
      break;

    case Token::kw_overlay:
      if (const Command *cmd = parseOverlay())
        sectionsCommands.push_back(cmd);
      else
        return nullptr;
      break;

    default:
      unrecognizedToken = true;
      break;
    }
  }

  if (!expectAndConsume(
          Token::r_brace,
          "expected symbol assignment, entry, overlay or output section name."))
    return nullptr;

  return new (_alloc) Sections(*this, sectionsCommands);
}

Memory *Parser::parseMemory() {
  assert(_tok._kind == Token::kw_memory && "Expected MEMORY!");
  consumeToken();
  if (!expectAndConsume(Token::l_brace, "expected {"))
    return nullptr;
  SmallVector<const MemoryBlock *, 8> blocks;

  bool unrecognizedToken = false;
  // Parse zero or more memory block descriptors.
  while (!unrecognizedToken) {
    if (_tok._kind == Token::identifier) {
      StringRef name;
      StringRef attrs;
      const Expression *origin = nullptr;
      const Expression *length = nullptr;

      name = _tok._range;
      consumeToken();

      // Parse optional memory region attributes.
      if (_tok._kind == Token::l_paren) {
        consumeToken();

        if (_tok._kind != Token::identifier) {
          error(_tok, "Expected memory attribute string.");
          return nullptr;
        }
        attrs = _tok._range;
        consumeToken();

        if (!expectAndConsume(Token::r_paren, "expected )"))
          return nullptr;
      }

      if (!expectAndConsume(Token::colon, "expected :"))
        return nullptr;

      // Parse the ORIGIN (base address of memory block).
      if (!expectAndConsume(Token::kw_origin, "expected ORIGIN"))
        return nullptr;

      if (!expectAndConsume(Token::equal, "expected ="))
        return nullptr;

      origin = parseExpression();
      if (!origin)
        return nullptr;

      if (!expectAndConsume(Token::comma, "expected ,"))
        return nullptr;

      // Parse the LENGTH (length of memory block).
      if (!expectAndConsume(Token::kw_length, "expected LENGTH"))
        return nullptr;

      if (!expectAndConsume(Token::equal, "expected ="))
        return nullptr;

      length = parseExpression();
      if (!length)
        return nullptr;

      auto *block = new (_alloc) MemoryBlock(name, attrs, origin, length);
      blocks.push_back(block);
    } else {
      unrecognizedToken = true;
    }
  }
  if (!expectAndConsume(
          Token::r_brace,
          "expected memory block definition."))
    return nullptr;

  return new (_alloc) Memory(*this, blocks);
}

Extern *Parser::parseExtern() {
  assert(_tok._kind == Token::kw_extern && "Expected EXTERN!");
  consumeToken();
  if (!expectAndConsume(Token::l_paren, "expected ("))
    return nullptr;

  // Parse one or more symbols.
  SmallVector<StringRef, 8> symbols;
  if (_tok._kind != Token::identifier) {
    error(_tok, "expected one or more symbols in EXTERN.");
    return nullptr;
  }
  symbols.push_back(_tok._range);
  consumeToken();
  while (_tok._kind == Token::identifier) {
    symbols.push_back(_tok._range);
    consumeToken();
  }

  if (!expectAndConsume(Token::r_paren, "expected symbol in EXTERN."))
    return nullptr;

  return new (_alloc) Extern(*this, symbols);
}

// Sema member functions
Sema::Sema() : _programPHDR(nullptr) {}

std::error_code Sema::perform() {
  llvm::StringMap<const PHDR *> phdrs;

  for (auto &parser : _scripts) {
    for (const Command *c : parser->get()->_commands) {
      if (const auto *sec = dyn_cast<Sections>(c)) {
        linearizeAST(sec);
      } else if (const auto *ph = dyn_cast<PHDRS>(c)) {
        if (auto ec = collectPHDRs(ph, phdrs))
          return ec;
      }
    }
  }
  return buildSectionToPHDR(phdrs);
}

bool Sema::less(const SectionKey &lhs, const SectionKey &rhs) const {
  int a = getLayoutOrder(lhs, true);
  int b = getLayoutOrder(rhs, true);

  if (a != b) {
    if (a < 0)
      return false;
    if (b < 0)
      return true;
    return a < b;
  }

  // If both sections are not mapped anywhere, they have the same order
  if (a < 0)
    return false;

  // If both sections fall into the same layout order, we need to find their
  // relative position as written in the (InputSectionsCmd).
  return localCompare(a, lhs, rhs);
}

StringRef Sema::getOutputSection(const SectionKey &key) const {
  int layoutOrder = getLayoutOrder(key, true);
  if (layoutOrder < 0)
    return StringRef();

  for (int i = layoutOrder - 1; i >= 0; --i) {
    if (!isa<OutputSectionDescription>(_layoutCommands[i]))
      continue;

    const OutputSectionDescription *out =
        dyn_cast<OutputSectionDescription>(_layoutCommands[i]);
    return out->name();
  }

  return StringRef();
}

std::vector<const SymbolAssignment *>
Sema::getExprs(const SectionKey &key) {
  int layoutOrder = getLayoutOrder(key, false);
  auto ans = std::vector<const SymbolAssignment *>();

  if (layoutOrder < 0 || _deliveredExprs.count(layoutOrder) > 0)
    return ans;

  for (int i = layoutOrder - 1; i >= 0; --i) {
    if (isa<InputSection>(_layoutCommands[i]))
      break;
    if (auto assgn = dyn_cast<SymbolAssignment>(_layoutCommands[i]))
      ans.push_back(assgn);
  }

  // Reverse this order so we evaluate the expressions in the original order
  // of the linker script
  std::reverse(ans.begin(), ans.end());

  // Mark this layout number as delivered
  _deliveredExprs.insert(layoutOrder);
  return ans;
}

std::error_code Sema::evalExpr(const SymbolAssignment *assgn,
                               uint64_t &curPos) {
  _symbolTable[StringRef(".")] = curPos;

  auto ans = assgn->expr()->evalExpr(_symbolTable);
  if (ans.getError())
    return ans.getError();
  uint64_t result = *ans;

  if (assgn->symbol() == ".") {
    curPos = result;
    return std::error_code();
  }

  _symbolTable[assgn->symbol()] = result;
  return std::error_code();
}

const llvm::StringSet<> &Sema::getScriptDefinedSymbols() const {
  // Do we have cached results?
  if (!_definedSymbols.empty())
    return _definedSymbols;

  // Populate our defined set and return it
  for (auto cmd : _layoutCommands)
    if (auto sa = dyn_cast<SymbolAssignment>(cmd)) {
      StringRef symbol = sa->symbol();
      if (!symbol.empty() && symbol != ".")
        _definedSymbols.insert(symbol);
    }

  return _definedSymbols;
}

uint64_t Sema::getLinkerScriptExprValue(StringRef name) const {
  auto it = _symbolTable.find(name);
  assert (it != _symbolTable.end() && "Invalid symbol name!");
  return it->second;
}

bool Sema::hasPHDRs() const { return !_sectionToPHDR.empty(); }

std::vector<const PHDR *> Sema::getPHDRsForOutputSection(StringRef name) const {
  auto vec = _sectionToPHDR.lookup(name);
  return std::vector<const PHDR *>(std::begin(vec), std::end(vec));
}

const PHDR *Sema::getProgramPHDR() const { return _programPHDR; }

void Sema::dump() const {
  raw_ostream &os = llvm::outs();
  os << "Linker script semantics dump\n";
  int num = 0;
  for (auto &parser : _scripts) {
    os << "Dumping script #" << ++num << ":\n";
    parser->get()->dump(os);
    os << "\n";
  }
  os << "Dumping rule ids:\n";
  for (unsigned i = 0; i < _layoutCommands.size(); ++i) {
    os << "LayoutOrder " << i << ":\n";
    _layoutCommands[i]->dump(os);
    os << "\n\n";
  }
}

/// Given a string "pattern" with wildcard characters, return true if it
/// matches "name". This function is useful when checking if a given name
/// pattern written in the linker script, i.e. ".text*", should match
/// ".text.anytext".
static bool wildcardMatch(StringRef pattern, StringRef name) {
  auto i = name.begin();

  // Check if each char in pattern also appears in our input name, handling
  // special wildcard characters.
  for (auto j = pattern.begin(), e = pattern.end(); j != e; ++j) {
    if (i == name.end())
      return false;

    switch (*j) {
    case '*':
      while (!wildcardMatch(pattern.drop_front(j - pattern.begin() + 1),
                            name.drop_front(i - name.begin()))) {
        if (i == name.end())
          return false;
        ++i;
      }
      break;
    case '?':
      // Matches any character
      ++i;
      break;
    case '[': {
      // Matches a range of characters specified between brackets
      size_t end = pattern.find(']', j - pattern.begin());
      if (end == pattern.size())
        return false;

      StringRef chars = pattern.slice(j - pattern.begin(), end);
      if (chars.find(i) == StringRef::npos)
        return false;

      j = pattern.begin() + end;
      ++i;
      break;
    }
    case '\\':
      ++j;
      if (*j != *i)
        return false;
      ++i;
      break;
    default:
      // No wildcard character means we must match exactly the same char
      if (*j != *i)
        return false;
      ++i;
      break;
    }
  }

  // If our pattern has't consumed the entire string, it is not a match
  return i == name.end();
}

int Sema::matchSectionName(int id, const SectionKey &key) const {
  const InputSectionsCmd *cmd = dyn_cast<InputSectionsCmd>(_layoutCommands[id]);

  if (!cmd || !wildcardMatch(cmd->archiveName(), key.archivePath))
    return -1;

  while ((size_t)++id < _layoutCommands.size() &&
         (isa<InputSection>(_layoutCommands[id]))) {
    if (isa<InputSectionSortedGroup>(_layoutCommands[id]))
      continue;

    const InputSectionName *in =
        dyn_cast<InputSectionName>(_layoutCommands[id]);
    if (wildcardMatch(in->name(), key.sectionName))
      return id;
  }
  return -1;
}

int Sema::getLayoutOrder(const SectionKey &key, bool coarse) const {
  // First check if we already answered this layout question
  if (coarse) {
    auto entry = _cacheSectionOrder.find(key);
    if (entry != _cacheSectionOrder.end())
      return entry->second;
  } else {
    auto entry = _cacheExpressionOrder.find(key);
    if (entry != _cacheExpressionOrder.end())
      return entry->second;
  }

  // Try to match exact file name
  auto range = _memberToLayoutOrder.equal_range(key.memberPath);
  for (auto I = range.first, E = range.second; I != E; ++I) {
    int order = I->second;
    int exprOrder = -1;

    if ((exprOrder = matchSectionName(order, key)) >= 0) {
      if (coarse) {
        _cacheSectionOrder.insert(std::make_pair(key, order));
        return order;
      }
      _cacheExpressionOrder.insert(std::make_pair(key, exprOrder));
      return exprOrder;
    }
  }

  // If we still couldn't find a rule for this input section, try to match
  // wildcards
  for (const auto &I : _memberNameWildcards) {
    if (!wildcardMatch(I.first, key.memberPath))
      continue;
    int order = I.second;
    int exprOrder = -1;

    if ((exprOrder = matchSectionName(order, key)) >= 0) {
      if (coarse) {
        _cacheSectionOrder.insert(std::make_pair(key, order));
        return order;
      }
      _cacheExpressionOrder.insert(std::make_pair(key, exprOrder));
      return exprOrder;
    }
  }

  _cacheSectionOrder.insert(std::make_pair(key, -1));
  _cacheExpressionOrder.insert(std::make_pair(key, -1));
  return -1;
}

static bool compareSortedNames(WildcardSortMode sortMode, StringRef lhs,
                               StringRef rhs) {
  switch (sortMode) {
  case WildcardSortMode::None:
  case WildcardSortMode::NA:
    return false;
  case WildcardSortMode::ByAlignment:
  case WildcardSortMode::ByInitPriority:
  case WildcardSortMode::ByAlignmentAndName:
    assert(false && "Unimplemented sort order");
    break;
  case WildcardSortMode::ByName:
    return lhs.compare(rhs) < 0;
  case WildcardSortMode::ByNameAndAlignment:
    int compare = lhs.compare(rhs);
    if (compare != 0)
      return compare < 0;
    return compareSortedNames(WildcardSortMode::ByAlignment, lhs, rhs);
  }
  return false;
}

static bool sortedGroupContains(const InputSectionSortedGroup *cmd,
                                const Sema::SectionKey &key) {
  for (const InputSection *child : *cmd) {
    if (auto i = dyn_cast<InputSectionName>(child)) {
      if (wildcardMatch(i->name(), key.sectionName))
        return true;
      continue;
    }

    auto *sortedGroup = dyn_cast<InputSectionSortedGroup>(child);
    assert(sortedGroup && "Expected InputSectionSortedGroup object");

    if (sortedGroupContains(sortedGroup, key))
      return true;
  }

  return false;
}

bool Sema::localCompare(int order, const SectionKey &lhs,
                        const SectionKey &rhs) const {
  const InputSectionsCmd *cmd =
      dyn_cast<InputSectionsCmd>(_layoutCommands[order]);

  assert(cmd && "Invalid InputSectionsCmd index");

  if (lhs.archivePath != rhs.archivePath)
    return compareSortedNames(cmd->archiveSortMode(), lhs.archivePath,
                              rhs.archivePath);

  if (lhs.memberPath != rhs.memberPath)
    return compareSortedNames(cmd->fileSortMode(), lhs.memberPath,
                              rhs.memberPath);

  // Both sections come from the same exact same file and rule. Start walking
  // through input section names as written in the linker script and the
  // first one to match will have higher priority.
  for (const InputSection *inputSection : *cmd) {
    if (auto i = dyn_cast<InputSectionName>(inputSection)) {
      // If both match, return false (both have equal priority)
      // If rhs match, return false (rhs has higher priority)
      if (wildcardMatch(i->name(), rhs.sectionName))
        return false;
      //  If lhs matches first, it has priority over rhs
      if (wildcardMatch(i->name(), lhs.sectionName))
        return true;
      continue;
    }

    // Handle sorted subgroups specially
    auto *sortedGroup = dyn_cast<InputSectionSortedGroup>(inputSection);
    assert(sortedGroup && "Expected InputSectionSortedGroup object");

    bool a = sortedGroupContains(sortedGroup, lhs);
    bool b = sortedGroupContains(sortedGroup, rhs);
    if (a && !b)
      return false;
    if (b && !a)
      return true;
    if (!a && !a)
      continue;

    return compareSortedNames(sortedGroup->sortMode(), lhs.sectionName,
                              rhs.sectionName);
  }

  llvm_unreachable("");
  return false;
}

std::error_code Sema::collectPHDRs(const PHDRS *ph,
                                   llvm::StringMap<const PHDR *> &phdrs) {
  bool loadFound = false;
  for (auto *p : *ph) {
    phdrs[p->name()] = p;

    switch (p->type()) {
    case llvm::ELF::PT_PHDR:
      if (_programPHDR != nullptr)
        return LinkerScriptReaderError::extra_program_phdr;
      if (loadFound)
        return LinkerScriptReaderError::misplaced_program_phdr;
      if (!p->hasPHDRs())
        return LinkerScriptReaderError::program_phdr_wrong_phdrs;
      _programPHDR = p;
      break;
    case llvm::ELF::PT_LOAD:
      // Program header, if available, should have program header table
      // mapped in the first loadable segment.
      if (!loadFound && _programPHDR && !p->hasPHDRs())
        return LinkerScriptReaderError::program_phdr_wrong_phdrs;
      loadFound = true;
      break;
    }
  }
  return std::error_code();
}

std::error_code Sema::buildSectionToPHDR(llvm::StringMap<const PHDR *> &phdrs) {
  const bool noPhdrs = phdrs.empty();

  // Add NONE header to the map provided there's no user-defined
  // header with the same name.
  if (!phdrs.count(PHDR_NONE.name()))
    phdrs[PHDR_NONE.name()] = &PHDR_NONE;

  // Match output sections to available headers.
  llvm::SmallVector<const PHDR *, 2> phdrsCur, phdrsLast { &PHDR_NONE };
  for (const Command *cmd : _layoutCommands) {
    auto osd = dyn_cast<OutputSectionDescription>(cmd);
    if (!osd || osd->isDiscarded())
      continue;

    phdrsCur.clear();
    for (StringRef name : osd->PHDRs()) {
      auto it = phdrs.find(name);
      if (it == phdrs.end()) {
        return LinkerScriptReaderError::unknown_phdr_ids;
      }
      phdrsCur.push_back(it->second);
    }

    // If no headers and no errors - insert empty headers set.
    // If the current set of headers is empty, then use the last non-empty
    // set. Otherwise mark the current set to be the last non-empty set for
    // successors.
    if (noPhdrs)
      phdrsCur.clear();
    else if (phdrsCur.empty())
      phdrsCur = phdrsLast;
    else
      phdrsLast = phdrsCur;

    _sectionToPHDR[osd->name()] = phdrsCur;
  }
  return std::error_code();
}

static bool hasWildcard(StringRef name) {
  for (auto ch : name)
    if (ch == '*' || ch == '?' || ch == '[' || ch == '\\')
      return true;
  return false;
}

void Sema::linearizeAST(const InputSection *inputSection) {
  if (isa<InputSectionName>(inputSection)) {
    _layoutCommands.push_back(inputSection);
    return;
  }

  auto *sortedGroup = dyn_cast<InputSectionSortedGroup>(inputSection);
  assert(sortedGroup && "Expected InputSectionSortedGroup object");

  for (const InputSection *child : *sortedGroup) {
    linearizeAST(child);
  }
}

void Sema::linearizeAST(const InputSectionsCmd *inputSections) {
  StringRef memberName = inputSections->memberName();
  // Populate our maps for fast lookup of InputSectionsCmd
  if (hasWildcard(memberName))
    _memberNameWildcards.push_back(
        std::make_pair(memberName, (int)_layoutCommands.size()));
  else if (!memberName.empty())
    _memberToLayoutOrder.insert(
        std::make_pair(memberName.str(), (int)_layoutCommands.size()));

  _layoutCommands.push_back(inputSections);
  for (const InputSection *inputSection : *inputSections)
    linearizeAST(inputSection);
}

void Sema::linearizeAST(const Sections *sections) {
  for (const Command *sectionCommand : *sections) {
    if (isa<SymbolAssignment>(sectionCommand)) {
      _layoutCommands.push_back(sectionCommand);
      continue;
    }

    if (!isa<OutputSectionDescription>(sectionCommand))
      continue;

    _layoutCommands.push_back(sectionCommand);
    auto *outSection = dyn_cast<OutputSectionDescription>(sectionCommand);

    for (const Command *outSecCommand : *outSection) {
      if (isa<SymbolAssignment>(outSecCommand)) {
        _layoutCommands.push_back(outSecCommand);
        continue;
      }

      if (!isa<InputSectionsCmd>(outSecCommand))
        continue;

      linearizeAST(dyn_cast<InputSectionsCmd>(outSecCommand));
    }
  }
}

} // end namespace script
} // end namespace lld
