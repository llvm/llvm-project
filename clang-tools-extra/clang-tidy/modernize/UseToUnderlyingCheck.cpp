//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseToUnderlyingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/FixIt.h"

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
      ReplacementFunctionHeader(Options.get("ReplacementFunctionHeader", "")),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {
  if (ReplacementFunction == "std::to_underlying") {
    if (ReplacementFunctionHeader.empty())
      ReplacementFunctionHeader = "<utility>";
    else if (ReplacementFunctionHeader != "<utility>")
      configurationDiag("'std::to_underlying' is declared in '<utility>', but "
                        "'ReplacementFunctionHeader' is set to '%0'")
          << ReplacementFunctionHeader;
  }
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
  Options.store(Opts, "ReplacementFunctionHeader", ReplacementFunctionHeader);
}

void UseToUnderlyingCheck::registerMatchers(MatchFinder *Finder) {
  // Match an explicit cast (``static_cast``, C-style or functional) from a
  // scoped enumeration to an integer type. Enum-to-enum casts and casts to a
  // floating-point or pointer type are excluded here; whether the conversion is
  // precise or imprecise is decided in check().
  Finder->addMatcher(
      explicitCastExpr(
          hasDestinationType(
              qualType(isInteger(), unless(hasCanonicalType(enumType())))),
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

  const QualType DestType = Cast->getType().getCanonicalType();
  const QualType UnderlyingType =
      Enum->getIntegerType().getCanonicalType().getUnqualifiedType();
  const QualType DestUnqualified = DestType.getUnqualifiedType();
  const bool IsPrecise = UnderlyingType == DestUnqualified;

  // A cast to ``bool`` that is not precise (i.e. the underlying type is not
  // ``bool``) expresses a truthiness test rather than a request for the
  // underlying value, so it is left untouched.
  if (DestType->isBooleanType() && !IsPrecise)
    return;

  if (!IsPrecise && ImpreciseCasts == ImpreciseCastsKind::Ignore)
    return;

  const auto Diag = diag(Cast->getBeginLoc(),
                         "use '%0' to convert a scoped enumeration to its "
                         "underlying type")
                    << ReplacementFunction;

  if (!IsPrecise && ImpreciseCasts == ImpreciseCastsKind::Warn)
    return;

  const StringRef OperandText =
      tooling::fixit::getText(*Operand, *Result.Context);
  if (OperandText.empty())
    return;

  const std::string Call =
      (ReplacementFunction + "(" + OperandText + ")").str();

  const bool ReplaceWholeCast =
      IsPrecise || ImpreciseCasts == ImpreciseCastsKind::UseUnderlyingType;
  if (ReplaceWholeCast) {
    // `static_cast<long>(e)` -> `to_underlying(e)`
    Diag << tooling::fixit::createReplacement(*Cast, Call);
  } else {
    // `static_cast<long>(e)` -> `static_cast<long>(to_underlying(e))`
    Diag << tooling::fixit::createReplacement(*Operand, Call);
  }

  if (!ReplacementFunctionHeader.empty())
    Diag << IncludeInserter.createIncludeInsertion(
        Result.SourceManager->getFileID(
            Result.SourceManager->getExpansionLoc(Cast->getBeginLoc())),
        ReplacementFunctionHeader);
}

} // namespace clang::tidy::modernize
