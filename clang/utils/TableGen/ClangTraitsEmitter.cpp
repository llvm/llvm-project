#include "TableGenBackends.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

namespace {

StringRef recordKindToMacro(const Record *R) {
  if (R->isSubClassOf("UnaryTrait"))
    return "TYPE_TRAIT_1";
  if (R->isSubClassOf("BinaryTrait"))
    return "TYPE_TRAIT_2";
  if (R->isSubClassOf("VariadicTrait"))
    return "TYPE_TRAIT_N";
  if (R->isSubClassOf("ArrayTrait"))
    return "ARRAY_TYPE_TRAIT";
  if (R->isSubClassOf("ExpressionTrait"))
    return "EXPRESSION_TRAIT";
  if (R->isSubClassOf("UnaryExprOrTypeTrait"))
    return "UNARY_EXPR_OR_TYPE_TRAIT";
  if (R->isSubClassOf("CXX11UnaryExprOrTypeTrait"))
    return "CXX11_UNARY_EXPR_OR_TYPE_TRAIT";
  if (R->isSubClassOf("TransformTypeTrait"))
    return "TRANSFORM_TYPE_TRAIT_DEF";

  llvm_unreachable("unexpected Trait subclass");
}

void emitMacro(const Record *R, raw_ostream &OS) {
  OS << recordKindToMacro(R) << "(";
  if (R->isSubClassOf("TransformTypeTrait")) {
    const StringRef StdName = R->getValueAsString("StdName");
    OS << R->getName() << ", " << StdName;
  } else {
    OS << R->getValueAsString("Spelling") << ", " << R->getName() << ", "
       << R->getValueAsString("KeyFlag");
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
  };

  for (const auto &[MacroName, MacroArgs] : MacroDefs) {
    OS << "#ifndef " << MacroName << "\n"
       << "#define " << MacroName << MacroArgs << "\n"
       << "#endif\n";
  }

  OS << '\n';

  for (const Record *R : Records.getAllDerivedDefinitions("Trait"))
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
  const auto UnaryTraits = Records.getAllDerivedDefinitions("UnaryTrait");
  const auto BinaryTraits = Records.getAllDerivedDefinitions("BinaryTrait");
  const auto VariadicTraits = Records.getAllDerivedDefinitions("VariadicTrait");

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

  const auto ArrayTraits = Records.getAllDerivedDefinitions("ArrayTrait");
  OS << "/// Names for the array type traits.\n"
        "enum ArrayTypeTrait {\n";
  emitEnumerators(OS, ArrayTraits);
  OS << "  ATT_Last = " << ArrayTraits.size() - 1
     << " // ATT_Last == last ATT\n"
     << "};\n\n";

  const auto UETTs = Records.getAllDerivedDefinitions("UnaryExprOrTypeTrait");
  const auto CXX11UETTs =
      Records.getAllDerivedDefinitions("CXX11UnaryExprOrTypeTrait");
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
  const auto UnaryTraits = Records.getAllDerivedDefinitions("UnaryTrait");
  const auto BinaryTraits = Records.getAllDerivedDefinitions("BinaryTrait");
  const auto VariadicTraits = Records.getAllDerivedDefinitions("VariadicTrait");

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
                        Records.getAllDerivedDefinitions("ArrayTrait"));
  emitNamesAndSpellings(
      OS, "UnaryExprOrTypeTrait",
      concat<const Record *const>(
          Records.getAllDerivedDefinitions("UnaryExprOrTypeTrait"),
          Records.getAllDerivedDefinitions("CXX11UnaryExprOrTypeTrait")));
}

void emitStdNameCases(const RecordKeeper &Records, raw_ostream &OS) {
  for (const Record *R : Records.getAllDerivedDefinitions("TypeTrait")) {
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
