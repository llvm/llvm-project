#include "clang/Support/BuiltinsUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/TableGen/Error.h"

void clang::ParseBuiltinType(llvm::StringRef T, llvm::StringRef Substitution,
                             std::string &Type, llvm::SMLoc *Loc) {
  assert(Loc);
  T = T.trim();
  if (T.consume_back("*")) {
    ParseBuiltinType(T, Substitution, Type, Loc);
    Type += "*";
  } else if (T.consume_back("const")) {
    ParseBuiltinType(T, Substitution, Type, Loc);
    Type += "C";
  } else if (T.consume_back("volatile")) {
    ParseBuiltinType(T, Substitution, Type, Loc);
    Type += "D";
  } else if (T.consume_back("restrict")) {
    ParseBuiltinType(T, Substitution, Type, Loc);
    Type += "R";
  } else if (T.consume_back("&")) {
    ParseBuiltinType(T, Substitution, Type, Loc);
    Type += "&";
  } else if (T.consume_front("long")) {
    Type += "L";
    ParseBuiltinType(T, Substitution, Type, Loc);
  } else if (T.consume_front("unsigned")) {
    Type += "U";
    ParseBuiltinType(T, Substitution, Type, Loc);
  } else if (T.consume_front("_Complex")) {
    Type += "X";
    ParseBuiltinType(T, Substitution, Type, Loc);
  } else if (T.consume_front("_Constant")) {
    Type += "I";
    ParseBuiltinType(T, Substitution, Type, Loc);
  } else if (T.consume_front("T")) {
    if (Substitution.empty())
      llvm::PrintFatalError(*Loc, "Not a template");
    ParseBuiltinType(Substitution, Substitution, Type, Loc);
  } else {
    auto ReturnTypeVal = llvm::StringSwitch<std::string>(T)
                             .Case("__builtin_va_list_ref", "A")
                             .Case("__builtin_va_list", "a")
                             .Case("__float128", "LLd")
                             .Case("__fp16", "h")
                             .Case("__int128_t", "LLLi")
                             .Case("_Float16", "x")
                             .Case("bool", "b")
                             .Case("char", "c")
                             .Case("constant_CFString", "F")
                             .Case("double", "d")
                             .Case("FILE", "P")
                             .Case("float", "f")
                             .Case("id", "G")
                             .Case("int", "i")
                             .Case("int32_t", "Zi")
                             .Case("int64_t", "Wi")
                             .Case("jmp_buf", "J")
                             .Case("msint32_t", "Ni")
                             .Case("msuint32_t", "UNi")
                             .Case("objc_super", "M")
                             .Case("pid_t", "p")
                             .Case("ptrdiff_t", "Y")
                             .Case("SEL", "H")
                             .Case("short", "s")
                             .Case("sigjmp_buf", "SJ")
                             .Case("size_t", "z")
                             .Case("ucontext_t", "K")
                             .Case("uint32_t", "UZi")
                             .Case("uint64_t", "UWi")
                             .Case("void", "v")
                             .Case("wchar_t", "w")
                             .Case("...", ".")
                             .Default("error");
    if (ReturnTypeVal == "error")
      llvm::PrintFatalError(*Loc, "Unknown Type: " + T);

    Type += ReturnTypeVal;
  }
}
