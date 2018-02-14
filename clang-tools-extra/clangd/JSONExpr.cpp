//=== JSONExpr.cpp - JSON expressions, parsing and serialization - C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "JSONExpr.h"
#include "llvm/Support/Format.h"
#include <cctype>

using namespace llvm;
namespace clang {
namespace clangd {
namespace json {

void Expr::copyFrom(const Expr &M) {
  Type = M.Type;
  switch (Type) {
  case T_Null:
  case T_Boolean:
  case T_Number:
    memcpy(Union.buffer, M.Union.buffer, sizeof(Union.buffer));
    break;
  case T_StringRef:
    create<StringRef>(M.as<StringRef>());
    break;
  case T_String:
    create<std::string>(M.as<std::string>());
    break;
  case T_Object:
    create<ObjectExpr>(M.as<ObjectExpr>());
    break;
  case T_Array:
    create<ArrayExpr>(M.as<ArrayExpr>());
    break;
  }
}

void Expr::moveFrom(const Expr &&M) {
  Type = M.Type;
  switch (Type) {
  case T_Null:
  case T_Boolean:
  case T_Number:
    memcpy(Union.buffer, M.Union.buffer, sizeof(Union.buffer));
    break;
  case T_StringRef:
    create<StringRef>(M.as<StringRef>());
    break;
  case T_String:
    create<std::string>(std::move(M.as<std::string>()));
    M.Type = T_Null;
    break;
  case T_Object:
    create<ObjectExpr>(std::move(M.as<ObjectExpr>()));
    M.Type = T_Null;
    break;
  case T_Array:
    create<ArrayExpr>(std::move(M.as<ArrayExpr>()));
    M.Type = T_Null;
    break;
  }
}

void Expr::destroy() {
  switch (Type) {
  case T_Null:
  case T_Boolean:
  case T_Number:
    break;
  case T_StringRef:
    as<StringRef>().~StringRef();
    break;
  case T_String:
    as<std::string>().~basic_string();
    break;
  case T_Object:
    as<ObjectExpr>().~ObjectExpr();
    break;
  case T_Array:
    as<ArrayExpr>().~ArrayExpr();
    break;
  }
}

namespace {
// Simple recursive-descent JSON parser.
class Parser {
public:
  Parser(StringRef JSON)
      : Start(JSON.begin()), P(JSON.begin()), End(JSON.end()) {}

  bool parseExpr(Expr &Out);

  bool assertEnd() {
    eatWhitespace();
    if (P == End)
      return true;
    return parseError("Text after end of document");
  }

  Error takeError() {
    assert(Err);
    return std::move(*Err);
  }

private:
  void eatWhitespace() {
    while (P != End && (*P == ' ' || *P == '\r' || *P == '\n' || *P == '\t'))
      ++P;
  }

  // On invalid syntax, parseX() functions return false and and set Err.
  bool parseNumber(char First, double &Out);
  bool parseString(std::string &Out);
  bool parseUnicode(std::string &Out);
  bool parseError(const char *Msg); // always returns false

  char next() { return P == End ? 0 : *P++; }
  char peek() { return P == End ? 0 : *P; }
  static bool isNumber(char C) {
    return C == '0' || C == '1' || C == '2' || C == '3' || C == '4' ||
           C == '5' || C == '6' || C == '7' || C == '8' || C == '9' ||
           C == 'e' || C == 'E' || C == '+' || C == '-' || C == '.';
  }
  static void encodeUtf8(uint32_t Rune, std::string &Out);

  Optional<Error> Err;
  const char *Start, *P, *End;
};

bool Parser::parseExpr(Expr &Out) {
  eatWhitespace();
  if (P == End)
    return parseError("Unexpected EOF");
  switch (char C = next()) {
  // Bare null/true/false are easy - first char identifies them.
  case 'n':
    Out = nullptr;
    return (next() == 'u' && next() == 'l' && next() == 'l') ||
           parseError("Invalid bareword");
  case 't':
    Out = true;
    return (next() == 'r' && next() == 'u' && next() == 'e') ||
           parseError("Invalid bareword");
  case 'f':
    Out = false;
    return (next() == 'a' && next() == 'l' && next() == 's' && next() == 'e') ||
           parseError("Invalid bareword");
  case '"': {
    std::string S;
    if (parseString(S)) {
      Out = std::move(S);
      return true;
    }
    return false;
  }
  case '[': {
    Out = json::ary{};
    json::ary &A = *Out.asArray();
    eatWhitespace();
    if (peek() == ']') {
      ++P;
      return true;
    }
    for (;;) {
      A.emplace_back(nullptr);
      if (!parseExpr(A.back()))
        return false;
      eatWhitespace();
      switch (next()) {
      case ',':
        eatWhitespace();
        continue;
      case ']':
        return true;
      default:
        return parseError("Expected , or ] after array element");
      }
    }
  }
  case '{': {
    Out = json::obj{};
    json::obj &O = *Out.asObject();
    eatWhitespace();
    if (peek() == '}') {
      ++P;
      return true;
    }
    for (;;) {
      if (next() != '"')
        return parseError("Expected object key");
      std::string K;
      if (!parseString(K))
        return false;
      eatWhitespace();
      if (next() != ':')
        return parseError("Expected : after object key");
      eatWhitespace();
      if (!parseExpr(O[std::move(K)]))
        return false;
      eatWhitespace();
      switch (next()) {
      case ',':
        eatWhitespace();
        continue;
      case '}':
        return true;
      default:
        return parseError("Expected , or } after object property");
      }
    }
  }
  default:
    if (isNumber(C)) {
      double Num;
      if (parseNumber(C, Num)) {
        Out = Num;
        return true;
      } else {
        return false;
      }
    }
    return parseError("Expected JSON value");
  }
}

bool Parser::parseNumber(char First, double &Out) {
  SmallString<24> S;
  S.push_back(First);
  while (isNumber(peek()))
    S.push_back(next());
  char *End;
  Out = std::strtod(S.c_str(), &End);
  return End == S.end() || parseError("Invalid number");
}

bool Parser::parseString(std::string &Out) {
  // leading quote was already consumed.
  for (char C = next(); C != '"'; C = next()) {
    if (LLVM_UNLIKELY(P == End))
      return parseError("Unterminated string");
    if (LLVM_UNLIKELY((C & 0x1f) == C))
      return parseError("Control character in string");
    if (LLVM_LIKELY(C != '\\')) {
      Out.push_back(C);
      continue;
    }
    // Handle escape sequence.
    switch (C = next()) {
    case '"':
    case '\\':
    case '/':
      Out.push_back(C);
      break;
    case 'b':
      Out.push_back('\b');
      break;
    case 'f':
      Out.push_back('\f');
      break;
    case 'n':
      Out.push_back('\n');
      break;
    case 'r':
      Out.push_back('\r');
      break;
    case 't':
      Out.push_back('\t');
      break;
    case 'u':
      if (!parseUnicode(Out))
        return false;
      break;
    default:
      return parseError("Invalid escape sequence");
    }
  }
  return true;
}

void Parser::encodeUtf8(uint32_t Rune, std::string &Out) {
  if (Rune <= 0x7F) {
    Out.push_back(Rune & 0x7F);
  } else if (Rune <= 0x7FF) {
    uint8_t FirstByte = 0xC0 | ((Rune & 0x7C0) >> 6);
    uint8_t SecondByte = 0x80 | (Rune & 0x3F);
    Out.push_back(FirstByte);
    Out.push_back(SecondByte);
  } else if (Rune <= 0xFFFF) {
    uint8_t FirstByte = 0xE0 | ((Rune & 0xF000) >> 12);
    uint8_t SecondByte = 0x80 | ((Rune & 0xFC0) >> 6);
    uint8_t ThirdByte = 0x80 | (Rune & 0x3F);
    Out.push_back(FirstByte);
    Out.push_back(SecondByte);
    Out.push_back(ThirdByte);
  } else if (Rune <= 0x10FFFF) {
    uint8_t FirstByte = 0xF0 | ((Rune & 0x1F0000) >> 18);
    uint8_t SecondByte = 0x80 | ((Rune & 0x3F000) >> 12);
    uint8_t ThirdByte = 0x80 | ((Rune & 0xFC0) >> 6);
    uint8_t FourthByte = 0x80 | (Rune & 0x3F);
    Out.push_back(FirstByte);
    Out.push_back(SecondByte);
    Out.push_back(ThirdByte);
    Out.push_back(FourthByte);
  } else {
    llvm_unreachable("Invalid codepoint");
  }
}

// Parse a \uNNNN escape sequence, the \u have already been consumed.
// May parse multiple escapes in the presence of surrogate pairs.
bool Parser::parseUnicode(std::string &Out) {
  // Note that invalid unicode is not a JSON error. It gets replaced by U+FFFD.
  auto Invalid = [&] { Out.append(/* UTF-8 */ {'\xef', '\xbf', '\xbd'}); };
  auto Parse4Hex = [this](uint16_t &Out) {
    Out = 0;
    char Bytes[] = {next(), next(), next(), next()};
    for (unsigned char C : Bytes) {
      if (!std::isxdigit(C))
        return parseError("Invalid \\u escape sequence");
      Out <<= 4;
      Out |= (C > '9') ? (C & ~0x20) - 'A' + 10 : (C - '0');
    }
    return true;
  };
  uint16_t First;
  if (!Parse4Hex(First))
    return false;

  // We loop to allow proper surrogate-pair error handling.
  while (true) {
    if (LLVM_LIKELY(First < 0xD800 || First >= 0xE000)) { // BMP.
      encodeUtf8(First, Out);
      return true;
    }

    if (First >= 0xDC00) {
      Invalid(); // Lone trailing surrogate.
      return true;
    }

    // We have a leading surrogate, and need a trailing one.
    // Don't advance P: a lone surrogate is valid JSON (but invalid unicode)
    if (P + 2 > End || *P != '\\' || *(P + 1) != 'u') {
      Invalid(); // Lone leading not followed by \u...
      return true;
    }
    P += 2;
    uint16_t Second;
    if (!Parse4Hex(Second))
      return false;
    if (Second < 0xDC00 || Second >= 0xE000) {
      Invalid();      // Leading surrogate not followed by trailing.
      First = Second; // Second escape still needs to be processed.
      continue;
    }

    // Valid surrogate pair.
    encodeUtf8(0x10000 | ((First - 0xD800) << 10) | (Second - 0xDC00), Out);
    return true;
  }
}

bool Parser::parseError(const char *Msg) {
  int Line = 1;
  const char *StartOfLine = Start;
  for (const char *X = Start; X < P; ++X) {
    if (*X == 0x0A) {
      ++Line;
      StartOfLine = X + 1;
    }
  }
  Err.emplace(
      llvm::make_unique<ParseError>(Msg, Line, P - StartOfLine, P - Start));
  return false;
}
} // namespace

Expected<Expr> parse(StringRef JSON) {
  Parser P(JSON);
  json::Expr E = nullptr;
  if (P.parseExpr(E))
    if (P.assertEnd())
      return std::move(E);
  return P.takeError();
}
char ParseError::ID = 0;

} // namespace json
} // namespace clangd
} // namespace clang

namespace {
void quote(llvm::raw_ostream &OS, llvm::StringRef S) {
  OS << '\"';
  for (unsigned char C : S) {
    if (C == 0x22 || C == 0x5C)
      OS << '\\';
    if (C >= 0x20) {
      OS << C;
      continue;
    }
    OS << '\\';
    switch (C) {
    // A few characters are common enough to make short escapes worthwhile.
    case '\t':
      OS << 't';
      break;
    case '\n':
      OS << 'n';
      break;
    case '\r':
      OS << 'r';
      break;
    default:
      OS << 'u';
      llvm::write_hex(OS, C, llvm::HexPrintStyle::Lower, 4);
      break;
    }
  }
  OS << '\"';
}

enum IndenterAction {
  Indent,
  Outdent,
  Newline,
  Space,
};
} // namespace

// Prints JSON. The indenter can be used to control formatting.
template <typename Indenter>
void clang::clangd::json::Expr::print(raw_ostream &OS,
                                      const Indenter &I) const {
  switch (Type) {
  case T_Null:
    OS << "null";
    break;
  case T_Boolean:
    OS << (as<bool>() ? "true" : "false");
    break;
  case T_Number:
    OS << format("%g", as<double>());
    break;
  case T_StringRef:
    quote(OS, as<StringRef>());
    break;
  case T_String:
    quote(OS, as<std::string>());
    break;
  case T_Object: {
    bool Comma = false;
    OS << '{';
    I(Indent);
    for (const auto &P : as<Expr::ObjectExpr>()) {
      if (Comma)
        OS << ',';
      Comma = true;
      I(Newline);
      quote(OS, P.first);
      OS << ':';
      I(Space);
      P.second.print(OS, I);
    }
    I(Outdent);
    if (Comma)
      I(Newline);
    OS << '}';
    break;
  }
  case T_Array: {
    bool Comma = false;
    OS << '[';
    I(Indent);
    for (const auto &E : as<Expr::ArrayExpr>()) {
      if (Comma)
        OS << ',';
      Comma = true;
      I(Newline);
      E.print(OS, I);
    }
    I(Outdent);
    if (Comma)
      I(Newline);
    OS << ']';
    break;
  }
  }
}

namespace clang {
namespace clangd {
namespace json {
llvm::raw_ostream &operator<<(raw_ostream &OS, const Expr &E) {
  E.print(OS, [](IndenterAction A) { /*ignore*/ });
  return OS;
}

bool operator==(const Expr &L, const Expr &R) {
  if (L.kind() != R.kind())
    return false;
  switch (L.kind()) {
  case Expr::Null:
    return *L.asNull() == *R.asNull();
  case Expr::Boolean:
    return *L.asBoolean() == *R.asBoolean();
  case Expr::Number:
    return *L.asNumber() == *R.asNumber();
  case Expr::String:
    return *L.asString() == *R.asString();
  case Expr::Array:
    return *L.asArray() == *R.asArray();
  case Expr::Object:
    return *L.asObject() == *R.asObject();
  }
  llvm_unreachable("Unknown expression kind");
}
} // namespace json
} // namespace clangd
} // namespace clang

void llvm::format_provider<clang::clangd::json::Expr>::format(
    const clang::clangd::json::Expr &E, raw_ostream &OS, StringRef Options) {
  if (Options.empty()) {
    OS << E;
    return;
  }
  unsigned IndentAmount = 0;
  if (Options.getAsInteger(/*Radix=*/10, IndentAmount))
    assert(false && "json::Expr format options should be an integer");
  unsigned IndentLevel = 0;
  E.print(OS, [&](IndenterAction A) {
    switch (A) {
    case Newline:
      OS << '\n';
      OS.indent(IndentLevel);
      break;
    case Space:
      OS << ' ';
      break;
    case Indent:
      IndentLevel += IndentAmount;
      break;
    case Outdent:
      IndentLevel -= IndentAmount;
      break;
    };
  });
}
