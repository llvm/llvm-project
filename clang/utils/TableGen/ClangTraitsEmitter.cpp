//==--- Traits.td - Generate expression and type traits -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {

std::vector<const Record *>
getAllDerivedDefsInDeclOrder(const RecordKeeper &Records, StringRef ClassName) {
  std::vector<const Record *> Defs =
      Records.getAllDerivedDefinitions(ClassName);
  llvm::sort(Defs, [](const Record *A, const Record *B) {
    return A->getID() < B->getID();
  });
  return Defs;
}

void emitKeyFlags(ArrayRef<const Record *> KeyFlags, raw_ostream &OS) {
  assert(!KeyFlags.empty() && "KeyFlags should never be empty");
  interleave(
      KeyFlags, OS, [&](const Record *KeyFlag) { OS << KeyFlag->getName(); },
      " | ");
}

StringRef recordKindToMacro(const Record *R) {
  auto const Macro =
      StringSwitch<StringRef>(R->getType()->getAsString())
          .Case("UnaryTrait", "TYPE_TRAIT_1")
          .Case("BinaryTrait", "TYPE_TRAIT_2")
          .Case("VariadicTrait", "TYPE_TRAIT_N")
          .Case("ArrayTrait", "ARRAY_TYPE_TRAIT")
          .Case("ExpressionTrait", "EXPRESSION_TRAIT")
          .Case("UnaryExprOrTypeTrait", "UNARY_EXPR_OR_TYPE_TRAIT")
          .Case("CXX11UnaryExprOrTypeTrait", "CXX11_UNARY_EXPR_OR_TYPE_TRAIT")
          .Case("TransformTypeTrait", "TRANSFORM_TYPE_TRAIT_DEF")
          .Case("Alias", "ALIAS")
          .Default("");
  assert(!Macro.empty() && "unexpected record class");
  return Macro;
}

void emitMacro(const Record *R, raw_ostream &OS) {
  OS << recordKindToMacro(R) << "(";
  if (R->isSubClassOf("TransformTypeTrait")) {
    const StringRef StdName = R->getValueAsString("StdName");
    OS << R->getName() << ", " << StdName;
  } else if (R->isSubClassOf("Alias")) {
    const Record *Primary = R->getValueAsDef("Primary");
    OS << "\"" << R->getValueAsString("Spelling") << "\", "
       << Primary->getValueAsString("Spelling") << ", ";
    emitKeyFlags(R->getValueAsListOfDefs("KeyFlags"), OS);
  } else {
    OS << R->getValueAsString("Spelling") << ", " << R->getName() << ", ";
    emitKeyFlags(R->getValueAsListOfDefs("KeyFlags"), OS);
  }
  OS << ")\n";
}

void emitMacroDefs(const RecordKeeper &Records, raw_ostream &OS) {
  constexpr std::pair<StringRef, StringRef> MacroDefs[] = {
      {"TYPE_TRAIT_1", "(I,E,K)"},
      {"TYPE_TRAIT_2", "(I,E,K)"},
      {"TYPE_TRAIT_N", "(I,E,K)"},
      {"ARRAY_TYPE_TRAIT", "(I,E,K)"},
      {"UNARY_EXPR_OR_TYPE_TRAIT", "(I,E,K)"},
      {"CXX11_UNARY_EXPR_OR_TYPE_TRAIT", "(I,E,K)"},
      {"EXPRESSION_TRAIT", "(I,E,K)"},
      {"TRANSFORM_TYPE_TRAIT_DEF", "(K, Trait)"},
      {"ALIAS", "(X,Y,Z)"}};

  for (const auto &[MacroName, MacroArgs] : MacroDefs) {
    OS << "#ifndef " << MacroName << "\n"
       << "#define " << MacroName << MacroArgs << "\n"
       << "#endif\n";
  }

  OS << '\n';

  const auto Traits = getAllDerivedDefsInDeclOrder(Records, "Trait");
  const auto Aliases = getAllDerivedDefsInDeclOrder(Records, "Alias");
  for (const Record *R : concat<const Record *const>(Traits, Aliases))
    emitMacro(R, OS);

  for (const auto &[MacroName, _] : reverse(MacroDefs))
    OS << "#undef " << MacroName << "\n";
}

template <typename RangeT>
void emitEnumerators(raw_ostream &OS, RangeT &&Range) {
  for (const Record *R : Range)
    OS << "  " << R->getValueAsString("Prefix") << '_' << R->getName() << ",\n";
}

void emitEnums(const RecordKeeper &Records, raw_ostream &OS) {
  const auto UnaryTraits = getAllDerivedDefsInDeclOrder(Records, "UnaryTrait");
  const auto BinaryTraits =
      getAllDerivedDefsInDeclOrder(Records, "BinaryTrait");
  const auto VariadicTraits =
      getAllDerivedDefsInDeclOrder(Records, "VariadicTrait");

  OS << "/// Names for traits that operate specifically on types.\n"
        "enum TypeTrait {\n";
  emitEnumerators(OS, UnaryTraits);
  OS << "  UTT_Last = " << UnaryTraits.size() - 1
     << ", // UTT_Last == last UTT_XX in the enum.\n";

  emitEnumerators(OS, BinaryTraits);
  OS << "  BTT_Last = " << UnaryTraits.size() + BinaryTraits.size() - 1
     << ", // BTT_Last == last BTT_XX in the enum.\n";

  emitEnumerators(OS, VariadicTraits);
  OS << "  TT_Last = "
     << UnaryTraits.size() + BinaryTraits.size() + VariadicTraits.size() - 1
     << " // TT_Last == last TT_XX in the enum.\n"
     << "};\n\n";

  const auto ArrayTraits = getAllDerivedDefsInDeclOrder(Records, "ArrayTrait");
  OS << "/// Names for the array type traits.\n"
        "enum ArrayTypeTrait {\n";
  emitEnumerators(OS, ArrayTraits);
  OS << "  ATT_Last = " << ArrayTraits.size() - 1
     << " // ATT_Last == last ATT\n"
     << "};\n\n";

  const auto UETTs =
      getAllDerivedDefsInDeclOrder(Records, "UnaryExprOrTypeTrait");
  const auto CXX11UETTs =
      getAllDerivedDefsInDeclOrder(Records, "CXX11UnaryExprOrTypeTrait");
  OS << "/// Names for the \"expression or type\" traits.\n"
        "enum UnaryExprOrTypeTrait {\n";
  emitEnumerators(OS, concat<const Record *const>(UETTs, CXX11UETTs));
  OS << "  UETT_Last = " << UETTs.size() + CXX11UETTs.size() - 1
     << " // UETT_Last == last UETT_XX in the enum.\n"
     << "};\n\n";
}

template <typename RangeT>
void emitNamesAndSpellings(raw_ostream &OS, StringRef Name, RangeT Range) {
  OS << "static constexpr const char *" << Name << "Names[] = {\n";
  for (const Record *R : Range) {
    OS << "  \"" << R->getName() << "\",\n";
  }
  OS << "};\n\n";

  OS << "static constexpr const char *" << Name << "Spellings[] = {\n";
  for (const Record *R : Range) {
    OS << "  \"" << R->getValueAsString("Spelling") << "\",\n";
  }
  OS << "};\n\n";
}

void emitArrays(const RecordKeeper &Records, raw_ostream &OS) {
  const auto UnaryTraits = getAllDerivedDefsInDeclOrder(Records, "UnaryTrait");
  const auto BinaryTraits =
      getAllDerivedDefsInDeclOrder(Records, "BinaryTrait");
  const auto VariadicTraits =
      getAllDerivedDefsInDeclOrder(Records, "VariadicTrait");

  emitNamesAndSpellings(
      OS, "TypeTrait",
      concat<const Record *const>(UnaryTraits, BinaryTraits, VariadicTraits));

  OS << "static constexpr const unsigned TypeTraitArities[] = {\n";
  interleaveComma(UnaryTraits, OS, [&](auto) { OS << '1'; });
  if (!UnaryTraits.empty())
    OS << ",\n";
  interleaveComma(BinaryTraits, OS, [&](auto) { OS << '2'; });
  if (!BinaryTraits.empty())
    OS << ",\n";
  interleaveComma(VariadicTraits, OS, [&](auto) { OS << '0'; });
  OS << "\n};\n\n";

  emitNamesAndSpellings(OS, "ArrayTypeTrait",
                        getAllDerivedDefsInDeclOrder(Records, "ArrayTrait"));
  emitNamesAndSpellings(
      OS, "UnaryExprOrTypeTrait",
      concat<const Record *const>(
          getAllDerivedDefsInDeclOrder(Records, "UnaryExprOrTypeTrait"),
          getAllDerivedDefsInDeclOrder(Records, "CXX11UnaryExprOrTypeTrait")));
}

void emitStdNameCases(const RecordKeeper &Records, raw_ostream &OS) {
  for (const Record *R : getAllDerivedDefsInDeclOrder(Records, "TypeTrait")) {
    const StringRef StdName = R->getValueAsString("StdName");
    if (StdName.empty())
      continue;

    OS << "  .Case(\"" << StdName
       << "\", TypeTrait::" << R->getValueAsString("Prefix") << '_'
       << R->getName() << ")\n";
  }
}

} // namespace

void clang::EmitClangTraits(const RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Type and expression traits", OS, Records);
  OS << "#if defined(EMIT_ENUMS)\n";
  emitEnums(Records, OS);

  OS << "#elif defined(EMIT_ARRAYS)\n";
  emitArrays(Records, OS);

  OS << "#elif defined(EMIT_STD_NAME_CASES)\n";
  emitStdNameCases(Records, OS);

  OS << "#else\n";
  emitMacroDefs(Records, OS);

  OS << "#endif\n\n"
     << R"(
#undef EMIT_ARRAYS
#undef EMIT_ENUMS
#undef EMIT_STD_NAME_CASES

)";
}
