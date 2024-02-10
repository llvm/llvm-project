//=- ClangBuiltinsEmitter.cpp - Generate Clang builtins tables -*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits Clang's builtins tables.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {
enum class BuiltinType {
  Builtin,
  AtomicBuiltin,
  LibBuiltin,
  LangBuiltin,
  TargetBuiltin,
};

class PrototypeParser {
public:
  PrototypeParser(StringRef Substitution, const Record *Builtin)
      : Loc(Builtin->getFieldLoc("Prototype")), Substitution(Substitution) {
    ParsePrototype(Builtin->getValueAsString("Prototype"));
  }

private:
  void ParsePrototype(StringRef Prototype) {
    Prototype = Prototype.trim();
    ParseTypes(Prototype);
  }

  void ParseTypes(StringRef &Prototype) {
    auto ReturnType = Prototype.take_until([](char c) { return c == '('; });
    ParseType(ReturnType);
    Prototype = Prototype.drop_front(ReturnType.size() + 1);
    if (!Prototype.ends_with(")"))
      PrintFatalError(Loc, "Expected closing brace at end of prototype");
    Prototype = Prototype.drop_back();
    for (auto T = Prototype.split(','); !T.first.empty();
         Prototype = T.second, T = Prototype.split(','))
      ParseType(T.first);
  }

  void ParseType(StringRef T) {
    T = T.trim();
    if (T.consume_back("*")) {
      ParseType(T);
      Type += "*";
    } else if (T.consume_back("const")) {
      ParseType(T);
      Type += "C";
    } else if (T.consume_back("volatile")) {
      ParseType(T);
      Type += "D";
    } else if (T.consume_back("restrict")) {
      ParseType(T);
      Type += "R";
    } else if (T.consume_back("&")) {
      ParseType(T);
      Type += "&";
    } else if (T.consume_front("long")) {
      Type += "L";
      ParseType(T);
    } else if (T.consume_front("unsigned")) {
      Type += "U";
      ParseType(T);
    } else if (T.consume_front("_Complex")) {
      Type += "X";
      ParseType(T);
    } else if (T.consume_front("_Constant")) {
      Type += "I";
      ParseType(T);
    } else if (T.consume_front("T")) {
      if (Substitution.empty())
        PrintFatalError(Loc, "Not a template");
      ParseType(Substitution);
    } else {
      auto ReturnTypeVal = StringSwitch<std::string>(T)
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
        PrintFatalError(Loc, "Unknown Type: " + T);
      Type += ReturnTypeVal;
    }
  }

public:
  void Print(llvm::raw_ostream &OS) const { OS << ", \"" << Type << '\"'; }

private:
  SMLoc Loc;
  StringRef Substitution;
  std::string Type;
};

class HeaderNameParser {
public:
  HeaderNameParser(const Record *Builtin) {
    for (char c : Builtin->getValueAsString("Header")) {
      if (std::islower(c))
        HeaderName += static_cast<char>(std::toupper(c));
      else if (c == '.' || c == '_' || c == '/' || c == '-')
        HeaderName += '_';
      else
        PrintFatalError(Builtin->getLoc(), "Unexpected header name");
    }
  }

  void Print(llvm::raw_ostream &OS) const { OS << HeaderName; }

private:
  std::string HeaderName;
};

void PrintAttributes(const Record *Builtin, BuiltinType BT,
                     llvm::raw_ostream &OS) {
  OS << '\"';
  if (Builtin->isSubClassOf("LibBuiltin")) {
    if (BT == BuiltinType::LibBuiltin) {
      OS << 'f';
    } else {
      OS << 'F';
      if (Builtin->getValueAsBit("OnlyBuiltinPrefixedAliasIsConstexpr"))
        OS << 'E';
    }
  }

  if (auto NS = Builtin->getValueAsOptionalString("Namespace")) {
    if (NS != "std")
      PrintFatalError(Builtin->getFieldLoc("Namespace"), "Unknown namespace: ");
    OS << "z";
  }

  for (const auto *Attr : Builtin->getValueAsListOfDefs("Attributes")) {
    OS << Attr->getValueAsString("Mangling");
    if (Attr->isSubClassOf("IndexedAttribute"))
      OS << ':' << Attr->getValueAsInt("Index") << ':';
  }
  OS << '\"';
}

void EmitBuiltinDef(llvm::raw_ostream &OS, StringRef Substitution,
                    const Record *Builtin, Twine Spelling, BuiltinType BT) {
  if (Builtin->getValueAsBit("RequiresUndef"))
    OS << "#undef " << Spelling << '\n';
  switch (BT) {
  case BuiltinType::LibBuiltin:
    OS << "LIBBUILTIN";
    break;
  case BuiltinType::LangBuiltin:
    OS << "LANGBUILTIN";
    break;
  case BuiltinType::Builtin:
    OS << "BUILTIN";
    break;
  case BuiltinType::AtomicBuiltin:
    OS << "ATOMIC_BUILTIN";
    break;
  case BuiltinType::TargetBuiltin:
    OS << "TARGET_BUILTIN";
    break;
  }

  OS << "(" << Spelling;
  PrototypeParser{Substitution, Builtin}.Print(OS);
  OS << ", ";
  PrintAttributes(Builtin, BT, OS);

  switch (BT) {
  case BuiltinType::LibBuiltin: {
    OS << ", ";
    HeaderNameParser{Builtin}.Print(OS);
    [[fallthrough]];
  }
  case BuiltinType::LangBuiltin: {
    OS << ", " << Builtin->getValueAsString("Languages");
    break;
  }
  case BuiltinType::TargetBuiltin:
    OS << ", \"" << Builtin->getValueAsString("Features") << "\"";
    break;
  case BuiltinType::AtomicBuiltin:
  case BuiltinType::Builtin:
    break;
  }
  OS << ")\n";
}

struct TemplateInsts {
  std::vector<std::string> Substitution;
  std::vector<std::string> Affix;
  bool IsPrefix;
};

TemplateInsts getTemplateInsts(const Record *R) {
  TemplateInsts temp;
  auto Substitutions = R->getValueAsListOfStrings("Substitutions");
  auto Affixes = R->getValueAsListOfStrings("Affixes");
  temp.IsPrefix = R->getValueAsBit("AsPrefix");

  if (Substitutions.size() != Affixes.size())
    PrintFatalError(R->getLoc(), "Substitutions and affixes "
                                 "don't have the same lengths");

  for (auto [Affix, Substitution] : llvm::zip(Affixes, Substitutions)) {
    temp.Substitution.emplace_back(Substitution);
    temp.Affix.emplace_back(Affix);
  }
  return temp;
}

void EmitBuiltin(llvm::raw_ostream &OS, const Record *Builtin) {
  TemplateInsts Templates = {};
  if (Builtin->isSubClassOf("Template")) {
    Templates = getTemplateInsts(Builtin);
  } else {
    Templates.Affix.emplace_back();
    Templates.Substitution.emplace_back();
  }

  for (auto [Substitution, Affix] :
       llvm::zip(Templates.Substitution, Templates.Affix)) {
    for (StringRef Spelling : Builtin->getValueAsListOfStrings("Spellings")) {
      auto FullSpelling =
          (Templates.IsPrefix ? Affix + Spelling : Spelling + Affix).str();
      BuiltinType BT = BuiltinType::Builtin;
      if (Builtin->isSubClassOf("AtomicBuiltin")) {
        BT = BuiltinType::AtomicBuiltin;
      } else if (Builtin->isSubClassOf("LangBuiltin")) {
        BT = BuiltinType::LangBuiltin;
      } else if (Builtin->isSubClassOf("TargetBuiltin")) {
        BT = BuiltinType::TargetBuiltin;
      } else if (Builtin->isSubClassOf("LibBuiltin")) {
        BT = BuiltinType::LibBuiltin;
        if (Builtin->getValueAsBit("AddBuiltinPrefixedAlias"))
          EmitBuiltinDef(OS, Substitution, Builtin,
                         std::string("__builtin_") + FullSpelling,
                         BuiltinType::Builtin);
      }
      EmitBuiltinDef(OS, Substitution, Builtin, FullSpelling, BT);
    }
  }
}
} // namespace

void clang::EmitClangBuiltins(llvm::RecordKeeper &Records,
                              llvm::raw_ostream &OS) {
  emitSourceFileHeader("List of builtins that Clang recognizes", OS);

  OS << R"c++(
#if defined(BUILTIN) && !defined(LIBBUILTIN)
#  define LIBBUILTIN(ID, TYPE, ATTRS, HEADER, BUILTIN_LANG) BUILTIN(ID, TYPE, ATTRS)
#endif

#if defined(BUILTIN) && !defined(LANGBUILTIN)
#  define LANGBUILTIN(ID, TYPE, ATTRS, BUILTIN_LANG) BUILTIN(ID, TYPE, ATTRS)
#endif

// Some of our atomics builtins are handled by AtomicExpr rather than
// as normal builtin CallExprs. This macro is used for such builtins.
#ifndef ATOMIC_BUILTIN
#  define ATOMIC_BUILTIN(ID, TYPE, ATTRS) BUILTIN(ID, TYPE, ATTRS)
#endif

#if defined(BUILTIN) && !defined(TARGET_BUILTIN)
#  define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE) BUILTIN(ID, TYPE, ATTRS)
#endif
)c++";

  // AtomicBuiltins are order dependent
  // emit them first to make manual checking easier
  for (const auto *Builtin : Records.getAllDerivedDefinitions("AtomicBuiltin"))
    EmitBuiltin(OS, Builtin);

  for (const auto *Builtin : Records.getAllDerivedDefinitions("Builtin")) {
    if (Builtin->isSubClassOf("AtomicBuiltin"))
      continue;
    EmitBuiltin(OS, Builtin);
  }

  for (const auto *Entry : Records.getAllDerivedDefinitions("CustomEntry")) {
    OS << Entry->getValueAsString("Entry") << '\n';
  }

  OS << R"c++(
#undef ATOMIC_BUILTIN
#undef BUILTIN
#undef LIBBUILTIN
#undef LANGBUILTIN
#undef TARGET_BUILTIN
)c++";
}
