#include "JSONExpr.h"

#include "llvm/Support/Format.h"

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
    create<Object>(M.as<Object>());
    break;
  case T_Array:
    create<Array>(M.as<Array>());
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
    create<Object>(std::move(M.as<Object>()));
    M.Type = T_Null;
    break;
  case T_Array:
    create<Array>(std::move(M.as<Array>()));
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
    as<Object>().~Object();
    break;
  case T_Array:
    as<Array>().~Array();
    break;
  }
}

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
    for (const auto &P : as<Expr::Object>()) {
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
    for (const auto &E : as<Expr::Array>()) {
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
