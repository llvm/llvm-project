//===--- UseToUnderlyingCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseToUnderlyingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy {

template <>
struct OptionEnumMapping<modernize::UseToUnderlyingCheck::ImpreciseCastsKind> {
  static llvm::ArrayRef<
      std::pair<modernize::UseToUnderlyingCheck::ImpreciseCastsKind, StringRef>>
  getEnumMapping() {
    using ImpreciseCastsKind =
        modernize::UseToUnderlyingCheck::ImpreciseCastsKind;
    static constexpr std::pair<ImpreciseCastsKind, StringRef> Mapping[] = {
        {ImpreciseCastsKind::Ignore, "Ignore"},
        {ImpreciseCastsKind::Warn, "Warn"},
        {ImpreciseCastsKind::PreserveType, "PreserveType"},
        {ImpreciseCastsKind::UseUnderlyingType, "UseUnderlyingType"},
    };
    return {Mapping};
  }
};

} // namespace clang::tidy

namespace clang::tidy::modernize {

UseToUnderlyingCheck::UseToUnderlyingCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ImpreciseCasts(Options.get("ImpreciseCasts", ImpreciseCastsKind::Warn)),
      ReplacementFunction(
          Options.get("ReplacementFunction", "std::to_underlying")),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      MaybeHeaderToInclude(Options.get("ReplacementFunctionHeader")) {
  if (!MaybeHeaderToInclude && ReplacementFunction == "std::to_underlying")
    MaybeHeaderToInclude = "<utility>";
}

bool UseToUnderlyingCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  // std::to_underlying is a C++23 library facility, but a user-provided
  // replacement (e.g. llvm::to_underlying) only requires scoped enumerations,
  // which are available since C++11.
  if (ReplacementFunction == "std::to_underlying")
    return LangOpts.CPlusPlus23;
  return LangOpts.CPlusPlus11;
}

void UseToUnderlyingCheck::registerPPCallbacks(const SourceManager &SM,
                                               Preprocessor *PP,
                                               Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseToUnderlyingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ImpreciseCasts", ImpreciseCasts);
  Options.store(Opts, "ReplacementFunction", ReplacementFunction);
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  if (MaybeHeaderToInclude)
    Options.store(Opts, "ReplacementFunctionHeader", *MaybeHeaderToInclude);
}

void UseToUnderlyingCheck::registerMatchers(MatchFinder *Finder) {
  // Match an explicit cast (``static_cast``, C-style or functional) whose
  // operand is an expression of scoped-enumeration type. The destination type
  // is validated in check() so that the same code path handles precise and
  // imprecise conversions.
  Finder->addMatcher(
      explicitCastExpr(
          unless(isInTemplateInstantiation()),
          hasSourceExpression(
              expr(hasType(hasCanonicalType(enumType(
                       hasDeclaration(enumDecl(isScoped()).bind("enum"))))))
                  .bind("operand")))
          .bind("cast"),
      this);
}

void UseToUnderlyingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Cast = Result.Nodes.getNodeAs<ExplicitCastExpr>("cast");
  const auto *Operand = Result.Nodes.getNodeAs<Expr>("operand");
  const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("enum");

  // Ignore dependent contexts we cannot reason about reliably.
  if (Cast->isValueDependent() || Cast->getType()->isDependentType())
    return;

  const QualType DestType = Cast->getType().getCanonicalType();

  // We only care about conversions to an integer type. Enum-to-enum casts and
  // floating-point or pointer destinations are excluded.
  if (!DestType->isIntegerType() || DestType->isEnumeralType())
    return;

  const QualType UnderlyingType =
      Enum->getIntegerType().getCanonicalType().getUnqualifiedType();
  const QualType DestUnqualified = DestType.getUnqualifiedType();
  const bool IsPrecise = UnderlyingType == DestUnqualified;

  // A cast to ``bool`` that is not precise (i.e. the underlying type is not
  // ``bool``) expresses a truthiness test rather than a request for the
  // underlying value, so it is left untouched.
  if (DestType->isBooleanType() && !IsPrecise)
    return;

  // Ignore imprecise casts if asked to do so.
  if (!IsPrecise && ImpreciseCasts == ImpreciseCastsKind::Ignore)
    return;

  auto Diag = diag(Cast->getBeginLoc(),
                   "use '%0' to convert a scoped enumeration to its "
                   "underlying type")
              << ReplacementFunction;

  // In Warn mode an imprecise cast is diagnosed without a fix-it
  if (!IsPrecise && ImpreciseCasts == ImpreciseCastsKind::Warn)
    return;

  const StringRef OperandText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Operand->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  if (OperandText.empty())
    return;

  const std::string Call =
      (ReplacementFunction + "(" + OperandText + ")").str();

  const bool ReplaceWholeCast =
      IsPrecise || ImpreciseCasts == ImpreciseCastsKind::UseUnderlyingType;
  if (ReplaceWholeCast) {
    // Replace the whole cast: ``static_cast<int>(e)`` -> ``to_underlying(e)``.
    // For an imprecise cast this changes the resulting type to the underlying
    // type.
    Diag << FixItHint::CreateReplacement(Cast->getSourceRange(), Call);
  } else {
    // Preserve the intended (wider or differently-signed) destination type by
    // wrapping only the operand: ``static_cast<long>(e)`` ->
    // ``static_cast<long>(to_underlying(e))``.
    Diag << FixItHint::CreateReplacement(Operand->getSourceRange(), Call);
  }

  if (MaybeHeaderToInclude)
    Diag << IncludeInserter.createIncludeInsertion(
        Result.SourceManager->getFileID(
            Result.SourceManager->getExpansionLoc(Cast->getBeginLoc())),
        *MaybeHeaderToInclude);
}

} // namespace clang::tidy::modernize
