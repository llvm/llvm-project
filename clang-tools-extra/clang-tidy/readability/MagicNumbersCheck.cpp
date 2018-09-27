//===--- MagicNumbersCheck.cpp - clang-tidy-------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A checker for magic numbers: integer or floating point literals embedded
// in the code, outside the definition of a constant or an enumeration.
//
//===----------------------------------------------------------------------===//

#include "MagicNumbersCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace clang::ast_matchers;
using namespace clang::ast_type_traits;

namespace {

bool isUsedToInitializeAConstant(const MatchFinder::MatchResult &Result,
                                 const DynTypedNode &Node) {

  const auto *AsDecl = Node.get<clang::DeclaratorDecl>();
  if (AsDecl) {
    if (AsDecl->getType().isConstQualified())
      return true;

    return AsDecl->isImplicit();
  }

  if (Node.get<clang::EnumConstantDecl>() != nullptr)
    return true;

  return llvm::any_of(Result.Context->getParents(Node),
                      [&Result](const DynTypedNode &Parent) {
                        return isUsedToInitializeAConstant(Result, Parent);
                      });
}

} // namespace

namespace clang {
namespace tidy {
namespace readability {

const char DefaultIgnoredIntegerValues[] = "1;2;3;4;";
const char DefaultIgnoredFloatingPointValues[] = "1.0;100.0;";

MagicNumbersCheck::MagicNumbersCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreAllFloatingPointValues(
          Options.get("IgnoreAllFloatingPointValues", false)),
      IgnorePowersOf2IntegerValues(
          Options.get("IgnorePowersOf2IntegerValues", false)) {
  // Process the set of ignored integer values.
  const std::vector<std::string> IgnoredIntegerValuesInput =
      utils::options::parseStringList(
          Options.get("IgnoredIntegerValues", DefaultIgnoredIntegerValues));
  IgnoredIntegerValues.resize(IgnoredIntegerValuesInput.size());
  llvm::transform(IgnoredIntegerValuesInput, IgnoredIntegerValues.begin(),
                  [](const std::string &Value) { return std::stoll(Value); });
  llvm::sort(IgnoredIntegerValues);

  if (!IgnoreAllFloatingPointValues) {
    // Process the set of ignored floating point values.
    const std::vector<std::string> IgnoredFloatingPointValuesInput =
        utils::options::parseStringList(Options.get(
            "IgnoredFloatingPointValues", DefaultIgnoredFloatingPointValues));
    IgnoredFloatingPointValues.reserve(IgnoredFloatingPointValuesInput.size());
    IgnoredDoublePointValues.reserve(IgnoredFloatingPointValuesInput.size());
    for (const auto &InputValue : IgnoredFloatingPointValuesInput) {
      llvm::APFloat FloatValue(llvm::APFloat::IEEEsingle());
      FloatValue.convertFromString(InputValue, DefaultRoundingMode);
      IgnoredFloatingPointValues.push_back(FloatValue.convertToFloat());

      llvm::APFloat DoubleValue(llvm::APFloat::IEEEdouble());
      DoubleValue.convertFromString(InputValue, DefaultRoundingMode);
      IgnoredDoublePointValues.push_back(DoubleValue.convertToDouble());
    }
    llvm::sort(IgnoredFloatingPointValues.begin(),
               IgnoredFloatingPointValues.end());
    llvm::sort(IgnoredDoublePointValues.begin(),
               IgnoredDoublePointValues.end());
  }
}

void MagicNumbersCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredIntegerValues", DefaultIgnoredIntegerValues);
  Options.store(Opts, "IgnoredFloatingPointValues",
                DefaultIgnoredFloatingPointValues);
}

void MagicNumbersCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(integerLiteral().bind("integer"), this);
  if (!IgnoreAllFloatingPointValues)
    Finder->addMatcher(floatLiteral().bind("float"), this);
}

void MagicNumbersCheck::check(const MatchFinder::MatchResult &Result) {
  checkBoundMatch<IntegerLiteral>(Result, "integer");
  checkBoundMatch<FloatingLiteral>(Result, "float");
}

bool MagicNumbersCheck::isConstant(const MatchFinder::MatchResult &Result,
                                   const Expr &ExprResult) const {
  return llvm::any_of(
      Result.Context->getParents(ExprResult),
      [&Result](const DynTypedNode &Parent) {
        return isUsedToInitializeAConstant(Result, Parent) ||
               // Ignore this instance, because this match reports the location
               // where the template is defined, not where it is instantiated.
               Parent.get<SubstNonTypeTemplateParmExpr>();
      });
}

bool MagicNumbersCheck::isIgnoredValue(const IntegerLiteral *Literal) const {
  const llvm::APInt IntValue = Literal->getValue();
  const int64_t Value = IntValue.getZExtValue();
  if (Value == 0)
    return true;

  if (IgnorePowersOf2IntegerValues && IntValue.isPowerOf2())
    return true;

  return std::binary_search(IgnoredIntegerValues.begin(),
                            IgnoredIntegerValues.end(), Value);
}

bool MagicNumbersCheck::isIgnoredValue(const FloatingLiteral *Literal) const {
  const llvm::APFloat FloatValue = Literal->getValue();
  if (FloatValue.isZero())
    return true;

  if (&FloatValue.getSemantics() == &llvm::APFloat::IEEEsingle()) {
    const float Value = FloatValue.convertToFloat();
    return std::binary_search(IgnoredFloatingPointValues.begin(),
                              IgnoredFloatingPointValues.end(), Value);
  }

  if (&FloatValue.getSemantics() == &llvm::APFloat::IEEEdouble()) {
    const double Value = FloatValue.convertToDouble();
    return std::binary_search(IgnoredDoublePointValues.begin(),
                              IgnoredDoublePointValues.end(), Value);
  }

  return false;
}

bool MagicNumbersCheck::isSyntheticValue(const SourceManager *SourceManager,
                                         const IntegerLiteral *Literal) const {
  const std::pair<FileID, unsigned> FileOffset =
      SourceManager->getDecomposedLoc(Literal->getLocation());
  if (FileOffset.first.isInvalid())
    return false;

  const StringRef BufferIdentifier =
      SourceManager->getBuffer(FileOffset.first)->getBufferIdentifier();

  return BufferIdentifier.empty();
}

} // namespace readability
} // namespace tidy
} // namespace clang
