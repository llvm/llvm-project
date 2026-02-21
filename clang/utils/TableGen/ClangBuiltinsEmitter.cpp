//===-- ClangBuiltinsEmitter.cpp - Generate Clang builtins tables ---------===//
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <sstream>

using namespace llvm;

namespace {
enum class BuiltinType {
  Builtin,
  AtomicBuiltin,
  LibBuiltin,
  LangBuiltin,
  TargetBuiltin,
  TargetLibBuiltin,
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

  void Print(raw_ostream &OS) const { OS << HeaderName; }

private:
  std::string HeaderName;
};

struct Builtin {
  BuiltinType BT;
  std::string Name;
  std::string Type;
  std::string Attributes;

  const Record *BuiltinRecord;

  void EmitEnumerator(llvm::raw_ostream &OS) const {
    OS << "    BI";
    // If there is a required name prefix, include its spelling in the
    // enumerator.
    if (auto *PrefixRecord =
            BuiltinRecord->getValueAsOptionalDef("RequiredNamePrefix"))
      OS << PrefixRecord->getValueAsString("Spelling");
    OS << Name << ",\n";
  }

  void EmitInfo(llvm::raw_ostream &OS, const StringToOffsetTable &Table) const {
    OS << "    Builtin::Info{Builtin::Info::StrOffsets{"
       << Table.GetStringOffset(Name) << " /* " << Name << " */, "
       << Table.GetStringOffset(Type) << " /* " << Type << " */, "
       << Table.GetStringOffset(Attributes) << " /* " << Attributes << " */, ";
    if (BT == BuiltinType::TargetBuiltin) {
      const auto &Features = BuiltinRecord->getValueAsString("Features");
      OS << Table.GetStringOffset(Features) << " /* " << Features << " */";
    } else {
      OS << "0";
    }
    OS << "}, ";
    if (BT == BuiltinType::LibBuiltin || BT == BuiltinType::TargetLibBuiltin) {
      OS << "HeaderDesc::";
      HeaderNameParser{BuiltinRecord}.Print(OS);
    } else {
      OS << "HeaderDesc::NO_HEADER";
    }
    OS << ", ";
    if (BT == BuiltinType::LibBuiltin || BT == BuiltinType::LangBuiltin ||
        BT == BuiltinType::TargetLibBuiltin) {
      OS << BuiltinRecord->getValueAsString("Languages");
    } else {
      OS << "ALL_LANGUAGES";
    }
    OS << "},\n";
  }

  void EmitXMacro(llvm::raw_ostream &OS) const {
    if (BuiltinRecord->getValueAsBit("RequiresUndef"))
      OS << "#undef " << Name << '\n';
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
    case BuiltinType::TargetLibBuiltin:
      OS << "TARGET_HEADER_BUILTIN";
      break;
    }

    OS << "(" << Name << ", \"" << Type << "\", \"" << Attributes << "\"";

    switch (BT) {
    case BuiltinType::LibBuiltin: {
      OS << ", ";
      HeaderNameParser{BuiltinRecord}.Print(OS);
      [[fallthrough]];
    }
    case BuiltinType::LangBuiltin: {
      OS << ", " << BuiltinRecord->getValueAsString("Languages");
      break;
    }
    case BuiltinType::TargetLibBuiltin: {
      OS << ", ";
      HeaderNameParser{BuiltinRecord}.Print(OS);
      OS << ", " << BuiltinRecord->getValueAsString("Languages");
      [[fallthrough]];
    }
    case BuiltinType::TargetBuiltin: {
      OS << ", \"" << BuiltinRecord->getValueAsString("Features") << "\"";
      break;
    }
    case BuiltinType::AtomicBuiltin:
    case BuiltinType::Builtin:
      break;
    }
    OS << ")\n";
  }
};

class PrototypeParser {
public:
  PrototypeParser(StringRef Substitution, const Record *Builtin)
      : Loc(Builtin->getFieldLoc("Prototype")), Substitution(Substitution),
        EnableOpenCLLong(Builtin->getValueAsBit("EnableOpenCLLong")) {
    ParsePrototype(Builtin->getValueAsString("Prototype"));
  }

  std::string takeTypeString() && { return std::move(Type); }

private:
  void ParsePrototype(StringRef Prototype) {
    Prototype = Prototype.trim();

    // Some builtins don't have an expressible prototype, simply emit an empty
    // string for them.
    if (Prototype.empty()) {
      Type = "";
      return;
    }

    ParseTypes(Prototype);
  }

  void ParseTypes(StringRef &Prototype) {
    auto ReturnType = Prototype.take_until([](char c) { return c == '('; });
    ParseType(ReturnType);
    Prototype = Prototype.drop_front(ReturnType.size() + 1);
    if (!Prototype.ends_with(")"))
      PrintFatalError(Loc, "Expected closing brace at end of prototype");
    Prototype = Prototype.drop_back();

    // Look through the input parameters.
    const size_t end = Prototype.size();
    for (size_t I = 0; I != end;) {
      const StringRef Current = Prototype.substr(I, end);
      // Skip any leading space or commas
      if (Current.starts_with(" ") || Current.starts_with(",")) {
        ++I;
        continue;
      }

      // Check if we are in _ExtVector. We do this first because
      // extended vectors are written in template form with the syntax
      // _ExtVector< ..., ...>, so we need to make sure we are not
      // detecting the comma of the template class as a separator for
      // the parameters of the prototype. Note: the assumption is that
      // we cannot have nested _ExtVector.
      if (Current.starts_with("_ExtVector<") ||
          Current.starts_with("_Vector<")) {
        size_t Pos = Current.find('<');
        int Depth = 1;

        // There may be a nested address_space<...> modifier on the type.
        while (Depth > 0 && ++Pos < Current.size()) {
          if (Current[Pos] == '<')
            ++Depth;
          else if (Current[Pos] == '>')
            --Depth;
        }

        const size_t EndTemplate = Pos;
        ParseType(Current.substr(0, EndTemplate + 1));
        // Move the prototype beyond _ExtVector<...>
        I += EndTemplate + 1;
        continue;
      }

      // We know that we are past _ExtVector, therefore the first seen
      // comma is the boundary of a parameter in the prototype.
      if (size_t CommaPos = Current.find(',', 0)) {
        if (CommaPos != StringRef::npos) {
          StringRef T = Current.substr(0, CommaPos);
          ParseType(T);
          // Move the prototype beyond the comma.
          I += CommaPos + 1;
          continue;
        }
      }

      // No more commas, parse final parameter.
      ParseType(Current);
      I = end;
    }
  }

  void ParseType(StringRef T) {
    T = T.trim();

    auto ConsumeAddrSpace = [&]() -> std::optional<unsigned> {
      T = T.trim();
      if (!T.consume_back(">"))
        return std::nullopt;

      auto Open = T.find_last_of('<');
      if (Open == StringRef::npos)
        PrintFatalError(Loc, "Mismatched angle-brackets in type");

      StringRef ArgStr = T.substr(Open + 1);
      T = T.slice(0, Open);
      if (!T.consume_back("address_space"))
        PrintFatalError(Loc,
                        "Only `address_space<N>` supported as a parameterized "
                        "pointer or reference type qualifier");

      unsigned Number = 0;
      if (ArgStr.getAsInteger(10, Number))
        PrintFatalError(
            Loc, "Expected an integer argument to the address_space qualifier");
      return Number;
    };

    if (T.consume_back("*")) {
      // Pointers may have an address space qualifier immediately before them.
      std::optional<unsigned> AS = ConsumeAddrSpace();
      // Pointers can apply to already parsed types, like vectors.
      if (!T.empty())
        ParseType(T);
      Type += "*";
      if (AS)
        Type += std::to_string(*AS);
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
      // References may have an address space qualifier immediately before them.
      std::optional<unsigned> AS = ConsumeAddrSpace();
      ParseType(T);
      Type += "&";
      if (AS)
        Type += std::to_string(*AS);
    } else if (T.consume_back(")")) {
      ParseType(T);
      Type += "&";
    } else if (EnableOpenCLLong && T.consume_front("long long")) {
      Type += "O";
      ParseType(T);
    } else if (T.consume_front("long")) {
      Type += "L";
      ParseType(T);
    } else if (T.consume_front("signed")) {
      Type += "S";
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
    } else if (auto IsExt = T.consume_front("_ExtVector");
               IsExt || T.consume_front("_Vector")) {
      // Clang extended vector types are mangled as follows:
      //
      // '_ExtVector<' <lanes> ',' <scalar type> '>'

      // Before parsing T(=<scalar type>), make sure the syntax of
      // `_ExtVector<N, T>` is correct...
      if (!T.consume_front("<"))
        PrintFatalError(Loc, "Expected '<' after '_ExtVector'");
      unsigned long long Lanes;
      if (consumeUnsignedInteger(T, 10, Lanes))
        PrintFatalError(Loc, "Expected number of lanes after '_ExtVector<'");
      Type += (IsExt ? "E" : "V") + std::to_string(Lanes);
      if (!T.consume_front(","))
        PrintFatalError(Loc,
                        "Expected ',' after number of lanes in '_ExtVector<'");
      if (!T.consume_back(">"))
        PrintFatalError(
            Loc, "Expected '>' after scalar type in '_ExtVector<N, type>'");

      // ...all good, we can check if we have a valid `<scalar type>`.
      ParseType(T);
    } else {
      auto ReturnTypeVal = StringSwitch<std::string>(T)
                               .Case("__builtin_va_list_ref", "A")
                               .Case("__builtin_va_list", "a")
                               .Case("__float128", "LLd")
                               .Case("__fp16", "h")
                               .Case("__hlsl_resource_t", "Qr")
                               .Case("__amdgpu_buffer_rsrc_t", "Qb")
                               .Case("__amdgpu_texture_t", "Qt")
                               .Case("__int128_t", "LLLi")
                               .Case("_Float16", "x")
                               .Case("__bf16", "y")
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

  SMLoc Loc;
  StringRef Substitution;
  bool EnableOpenCLLong;
  std::string Type;
};

std::string renderAttributes(const Record *Builtin, BuiltinType BT) {
  std::string Attributes;
  raw_string_ostream OS(Attributes);
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
    if (Attr->isSubClassOf("IndexedAttribute")) {
      OS << ':' << Attr->getValueAsInt("Index") << ':';
    } else if (Attr->isSubClassOf("MultiIndexAttribute")) {
      OS << '<';
      llvm::ListSeparator Sep(",");
      for (int64_t Index : Attr->getValueAsListOfInts("Indices"))
        OS << Sep << Index;
      OS << '>';
    }
  }
  return Attributes;
}

Builtin buildBuiltin(StringRef Substitution, const Record *BuiltinRecord,
                     Twine Spelling, BuiltinType BT) {
  Builtin B;
  B.BT = BT;
  B.Name = Spelling.str();
  B.Type = PrototypeParser(Substitution, BuiltinRecord).takeTypeString();
  B.Attributes = renderAttributes(BuiltinRecord, BT);
  B.BuiltinRecord = BuiltinRecord;
  return B;
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

  for (auto [Affix, Substitution] : zip(Affixes, Substitutions)) {
    temp.Substitution.emplace_back(Substitution);
    temp.Affix.emplace_back(Affix);
  }
  return temp;
}

void collectBuiltins(const Record *BuiltinRecord,
                     SmallVectorImpl<Builtin> &Builtins) {
  TemplateInsts Templates = {};
  if (BuiltinRecord->isSubClassOf("Template")) {
    Templates = getTemplateInsts(BuiltinRecord);
  } else {
    Templates.Affix.emplace_back();
    Templates.Substitution.emplace_back();
  }

  for (auto [Substitution, Affix] :
       zip(Templates.Substitution, Templates.Affix)) {
    for (StringRef Spelling :
         BuiltinRecord->getValueAsListOfStrings("Spellings")) {
      auto FullSpelling =
          (Templates.IsPrefix ? Affix + Spelling : Spelling + Affix).str();
      BuiltinType BT = BuiltinType::Builtin;
      if (BuiltinRecord->isSubClassOf("AtomicBuiltin")) {
        BT = BuiltinType::AtomicBuiltin;
      } else if (BuiltinRecord->isSubClassOf("LangBuiltin")) {
        BT = BuiltinType::LangBuiltin;
      } else if (BuiltinRecord->isSubClassOf("TargetLibBuiltin")) {
        BT = BuiltinType::TargetLibBuiltin;
      } else if (BuiltinRecord->isSubClassOf("TargetBuiltin")) {
        BT = BuiltinType::TargetBuiltin;
      } else if (BuiltinRecord->isSubClassOf("LibBuiltin")) {
        BT = BuiltinType::LibBuiltin;
        if (BuiltinRecord->getValueAsBit("AddBuiltinPrefixedAlias"))
          Builtins.push_back(buildBuiltin(
              Substitution, BuiltinRecord,
              std::string("__builtin_") + FullSpelling, BuiltinType::Builtin));
      }
      Builtins.push_back(
          buildBuiltin(Substitution, BuiltinRecord, FullSpelling, BT));
    }
  }
}
} // namespace

void clang::EmitClangBuiltins(const RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("List of builtins that Clang recognizes", OS);

  SmallVector<Builtin> Builtins;
  // AtomicBuiltins are order dependent. Emit them first to make manual checking
  // easier and so we can build a special atomic builtin X-macro.
  for (const auto *BuiltinRecord :
       Records.getAllDerivedDefinitions("AtomicBuiltin"))
    collectBuiltins(BuiltinRecord, Builtins);
  unsigned NumAtomicBuiltins = Builtins.size();

  for (const auto *BuiltinRecord :
       Records.getAllDerivedDefinitions("Builtin")) {
    if (BuiltinRecord->isSubClassOf("AtomicBuiltin"))
      continue;
    // Prefixed builtins are also special and we emit them last so they can have
    // their own representation that skips the prefix.
    if (BuiltinRecord->getValueAsOptionalDef("RequiredNamePrefix"))
      continue;

    collectBuiltins(BuiltinRecord, Builtins);
  }

  // Now collect (and count) the prefixed builtins.
  unsigned NumPrefixedBuiltins = Builtins.size();
  const Record *FirstPrefix = nullptr;
  for (const auto *BuiltinRecord :
       Records.getAllDerivedDefinitions("Builtin")) {
    auto *Prefix = BuiltinRecord->getValueAsOptionalDef("RequiredNamePrefix");
    if (!Prefix)
      continue;

    if (!FirstPrefix)
      FirstPrefix = Prefix;
    assert(Prefix == FirstPrefix &&
           "Multiple distinct prefixes which is not currently supported!");
    assert(!BuiltinRecord->isSubClassOf("AtomicBuiltin") &&
           "Cannot require a name prefix for an atomic builtin.");
    collectBuiltins(BuiltinRecord, Builtins);
  }
  NumPrefixedBuiltins = Builtins.size() - NumPrefixedBuiltins;

  auto AtomicBuiltins = ArrayRef(Builtins).slice(0, NumAtomicBuiltins);
  auto UnprefixedBuiltins = ArrayRef(Builtins).drop_back(NumPrefixedBuiltins);
  auto PrefixedBuiltins = ArrayRef(Builtins).take_back(NumPrefixedBuiltins);

  // Collect strings into a table.
  StringToOffsetTable Table;
  Table.GetOrAddStringOffset("");
  for (const auto &B : Builtins) {
    Table.GetOrAddStringOffset(B.Name);
    Table.GetOrAddStringOffset(B.Type);
    Table.GetOrAddStringOffset(B.Attributes);
    if (B.BT == BuiltinType::TargetBuiltin)
      Table.GetOrAddStringOffset(B.BuiltinRecord->getValueAsString("Features"));
  }

  // Emit enumerators.
  OS << R"c++(
#ifdef GET_BUILTIN_ENUMERATORS
)c++";
  for (const auto &B : Builtins)
    B.EmitEnumerator(OS);
  OS << R"c++(
#endif // GET_BUILTIN_ENUMERATORS
)c++";

  // Emit a string table that can be referenced for these builtins.
  OS << R"c++(
#ifdef GET_BUILTIN_STR_TABLE
)c++";
  Table.EmitStringTableDef(OS, "BuiltinStrings");
  OS << R"c++(
#endif // GET_BUILTIN_STR_TABLE
)c++";

  // Emit a direct set of `Builtin::Info` initializers, first for the unprefixed
  // builtins and then for the prefixed builtins.
  OS << R"c++(
#ifdef GET_BUILTIN_INFOS
)c++";
  for (const auto &B : UnprefixedBuiltins)
    B.EmitInfo(OS, Table);
  OS << R"c++(
#endif // GET_BUILTIN_INFOS
)c++";

  OS << R"c++(
#ifdef GET_BUILTIN_PREFIXED_INFOS
)c++";
  for (const auto &B : PrefixedBuiltins)
    B.EmitInfo(OS, Table);
  OS << R"c++(
#endif // GET_BUILTIN_PREFIXED_INFOS
)c++";

  // Emit X-macros for the atomic builtins to support various custome patterns
  // used exclusively with those builtins.
  //
  // FIXME: We should eventually move this to a separate file so that users
  // don't need to include the full set of builtins.
  OS << R"c++(
#ifdef ATOMIC_BUILTIN
)c++";
  for (const auto &Builtin : AtomicBuiltins) {
    Builtin.EmitXMacro(OS);
  }
  OS << R"c++(
#endif // ATOMIC_BUILTIN
#undef ATOMIC_BUILTIN
)c++";
}

//===----------------------------------------------------------------------===//
// Builtin documentation emitter
//===----------------------------------------------------------------------===//

/// Holds the data needed to emit documentation for a single builtin.
namespace {
struct BuiltinDocData {
  const Record *Documentation = nullptr;
  const Record *BuiltinRecord = nullptr;
  std::string Heading;

  BuiltinDocData(const Record *D, const Record *B)
      : Documentation(D), BuiltinRecord(B) {
    // Use the Heading field if set, otherwise use the builtin's first
    // spelling.
    StringRef HeadingStr = D->getValueAsString("Heading");
    if (HeadingStr.empty()) {
      std::vector<StringRef> Spellings =
          B->getValueAsListOfStrings("Spellings");
      if (!Spellings.empty())
        Heading = Spellings[0].str();
      else
        Heading = B->getName().str();
    } else {
      Heading = HeadingStr.str();
    }
  }
};
} // namespace

static void writeCategoryHeader(const Record *Category, raw_ostream &OS) {
  StringRef CategoryName = Category->getValueAsString("Name");
  OS << "\n" << CategoryName << "\n";
  for (size_t I = 0, E = CategoryName.size(); I < E; ++I)
    OS << "=";
  OS << "\n\n";

  StringRef CategoryContent = Category->getValueAsString("Content");
  if (!CategoryContent.trim().empty())
    OS << CategoryContent.trim() << "\n\n";
}

/// Split a parameter list string into individual parameter type strings,
/// respecting nested angle brackets (e.g. address_space<4>, _ExtVector<4,
/// float>).
static SmallVector<StringRef> splitParams(StringRef Params) {
  SmallVector<StringRef> Result;
  if (Params.empty())
    return Result;

  int Depth = 0;
  size_t Start = 0;
  for (size_t I = 0, E = Params.size(); I < E; ++I) {
    if (Params[I] == '<') {
      ++Depth;
    } else if (Params[I] == '>') {
      --Depth;
    } else if (Params[I] == ',' && Depth == 0) {
      Result.push_back(Params.substr(Start, I - Start).trim());
      Start = I + 1;
    }
  }
  // Add the last parameter.
  StringRef Last = Params.substr(Start).trim();
  if (!Last.empty())
    Result.push_back(Last);
  return Result;
}

static void writeBuiltinDocumentation(const BuiltinDocData &Doc,
                                      raw_ostream &OS) {
  // Write heading with '-' underline (subsection).
  std::string HeadingText = "``" + Doc.Heading + "``";
  OS << HeadingText << "\n";
  for (size_t I = 0, E = HeadingText.size(); I < E; ++I)
    OS << "-";
  OS << "\n\n";

  // Write prototype as a code block.
  StringRef Prototype = Doc.BuiltinRecord->getValueAsString("Prototype");
  if (!Prototype.empty()) {
    std::vector<StringRef> Spellings =
        Doc.BuiltinRecord->getValueAsListOfStrings("Spellings");
    StringRef Name =
        Spellings.empty() ? Doc.BuiltinRecord->getName() : Spellings[0];

    // Split prototype into return type and params at the first '('.
    size_t ParenPos = Prototype.find('(');
    if (ParenPos != StringRef::npos) {
      StringRef RetType = Prototype.substr(0, ParenPos).rtrim();
      StringRef ParamStr =
          Prototype.substr(ParenPos + 1, Prototype.size() - ParenPos - 2);

      OS << "**Prototype:**\n\n";
      OS << ".. code-block:: c\n\n";
      OS << "  " << RetType << " " << Name << "(";

      std::vector<StringRef> ArgNames =
          Doc.BuiltinRecord->getValueAsListOfStrings("ArgNames");
      if (!ArgNames.empty()) {
        SmallVector<StringRef> ParamTypes = splitParams(ParamStr);
        bool IsVariadic = !ParamTypes.empty() && ParamTypes.back() == "...";
        size_t NamedParams = ParamTypes.size() - (IsVariadic ? 1 : 0);
        if (ArgNames.size() != NamedParams)
          PrintFatalError(Doc.BuiltinRecord->getLoc(),
                          "number of ArgNames (" + Twine(ArgNames.size()) +
                              ") does not match number of prototype "
                              "parameters (" +
                              Twine(NamedParams) + ")");
        for (size_t I = 0, E = ParamTypes.size(); I < E; ++I) {
          if (I > 0)
            OS << ", ";
          if (I < NamedParams)
            OS << ParamTypes[I] << " " << ArgNames[I];
          else
            OS << ParamTypes[I];
        }
      } else {
        OS << ParamStr;
      }

      OS << ")\n\n";
    }
  }

  // Write target features if this is a TargetBuiltin with features.
  if (Doc.BuiltinRecord->isSubClassOf("TargetBuiltin")) {
    StringRef Features = Doc.BuiltinRecord->getValueAsString("Features");
    if (!Features.empty())
      OS << "**Target Features:** " << Features << "\n\n";
  }

  // Write documentation content.
  StringRef Content = Doc.Documentation->getValueAsString("Content");
  OS << Content.trim() << "\n\n\n";
}

void clang::EmitClangBuiltinDocs(const RecordKeeper &Records, raw_ostream &OS) {
  // Get the documentation introduction paragraph.
  const Record *Doc = Records.getDef("GlobalDocumentation");
  if (!Doc) {
    PrintFatalError("The GlobalDocumentation top-level definition is missing, "
                    "no documentation will be generated.");
  }

  OS << Doc->getValueAsString("Intro") << "\n";

  // Gather documentation from each builtin, grouped by category.
  llvm::MapVector<const Record *, std::vector<BuiltinDocData>> SplitDocs;

  for (const Record *B : Records.getAllDerivedDefinitions("Builtin")) {
    for (const Record *D : B->getValueAsListOfDefs("Documentation")) {
      const Record *Category = D->getValueAsDef("Category");
      StringRef Cat = Category->getValueAsString("Name");
      // Skip builtins that are explicitly internal-only.
      if (Cat == "InternalOnly")
        continue;
      SplitDocs[Category].emplace_back(D, B);
    }
  }

  // Sort categories alphabetically by name for deterministic output.
  llvm::sort(SplitDocs, [](const auto &A, const auto &B) {
    return A.first->getValueAsString("Name") <
           B.first->getValueAsString("Name");
  });

  // Write out each category and its builtins.
  for (auto &[Category, Docs] : SplitDocs) {
    writeCategoryHeader(Category, OS);

    // Sort entries alphabetically by heading.
    llvm::sort(Docs, [](const BuiltinDocData &A, const BuiltinDocData &B) {
      return A.Heading < B.Heading;
    });

    for (const BuiltinDocData &D : Docs)
      writeBuiltinDocumentation(D, OS);
  }
}
