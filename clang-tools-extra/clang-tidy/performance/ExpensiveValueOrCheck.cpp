//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpensiveValueOrCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {

bool hasOperatorStar(const CXXRecordDecl *RD) {
  return llvm::any_of(RD->methods(), [](const CXXMethodDecl *M) {
    return M->getOverloadedOperator() == OO_Star;
  });
}

StringRef findValueMethod(const CXXRecordDecl *RD) {
  for (const auto *M : RD->methods()) {
    if (!M->getDeclName().isIdentifier())
      continue;
    StringRef Name = M->getName();
    if (Name == "value" || Name == "Value")
      return Name;
  }
  return {};
}

std::string buildSuggestion(const CXXRecordDecl *OptionalClass) {
  bool HasDeref = hasOperatorStar(OptionalClass);
  StringRef ValueName = findValueMethod(OptionalClass);

  if (HasDeref && !ValueName.empty())
    return (llvm::Twine("consider using 'operator*' or '") + ValueName +
            "()' with a separate fallback")
        .str();
  if (HasDeref)
    return "consider using 'operator*' with a separate fallback";
  if (!ValueName.empty())
    return (llvm::Twine("consider using '") + ValueName +
            "()' with a separate fallback")
        .str();
  return "consider avoiding the copy";
}

std::optional<FixItHint>
buildFixIt(const CXXMemberCallExpr *Call, const Expr *ObjExpr,
           const Expr *FallbackArg, const CXXRecordDecl *OptionalClass,
           const SourceManager &SM, const LangOptions &LO) {
  if (Call->getBeginLoc().isMacroID())
    return std::nullopt;
  if (!ObjExpr->isLValue())
    return std::nullopt;
  if (!hasOperatorStar(OptionalClass))
    return std::nullopt;

  StringRef ObjText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(ObjExpr->getSourceRange()), SM, LO);
  StringRef ArgText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(FallbackArg->getSourceRange()), SM, LO);

  if (ObjText.empty() || ArgText.empty())
    return std::nullopt;

  std::string Replacement =
      ("(" + ObjText + " ? *" + ObjText + " : " + ArgText + ")").str();
  return FixItHint::CreateReplacement(Call->getSourceRange(), Replacement);
}

} // namespace

ExpensiveValueOrCheck::ExpensiveValueOrCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SizeThreshold(Options.get("SizeThreshold", 16U)),
      OptionalTypes(utils::options::parseStringList(
          Options.get("OptionalTypes",
                      "::std::optional;::absl::optional;::boost::optional"))),
      WarnOnOwnershipTaking(Options.get("WarnOnOwnershipTaking", false)) {}

void ExpensiveValueOrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SizeThreshold", SizeThreshold);
  Options.store(Opts, "OptionalTypes",
                utils::options::serializeStringList(OptionalTypes));
  Options.store(Opts, "WarnOnOwnershipTaking", WarnOnOwnershipTaking);
}

void ExpensiveValueOrCheck::registerMatchers(MatchFinder *Finder) {
  auto OptionalTypesMatcher =
      matchers::matchesAnyListedRegexName(OptionalTypes);
  auto ValueOrMatcher = hasAnyName("value_or", "valueOr", "ValueOr");
  auto ValueOrCall = cxxMemberCallExpr(
      callee(cxxMethodDecl(ValueOrMatcher, ofClass(OptionalTypesMatcher))));

  if (WarnOnOwnershipTaking) {
    Finder->addMatcher(ValueOrCall.bind("call"), this);
    return;
  }

  // Binding to const T& variable.
  Finder->addMatcher(
      varDecl(hasType(lValueReferenceType(pointee(isConstQualified()))),
              hasInitializer(ignoringImplicit(ValueOrCall.bind("call")))),
      this);

  // Passing to a const T& parameter.
  Finder->addMatcher(callExpr(forEachArgumentWithParam(
                         ignoringImplicit(ValueOrCall.bind("call")),
                         parmVarDecl(hasType(lValueReferenceType(
                             pointee(isConstQualified())))))),
                     this);

  // Calling a const member function on the result.
  Finder->addMatcher(
      cxxMemberCallExpr(on(ignoringImplicit(ValueOrCall.bind("call"))),
                        callee(cxxMethodDecl(isConst()))),
      this);
}

void ExpensiveValueOrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  assert(Call && "Matcher guaranteed a bound 'call' node");
  const Expr *ObjExpr = Call->getImplicitObjectArgument();
  assert(ObjExpr && "CXXMemberCallExpr must have an implicit object argument");

  const ASTContext &Ctx = *Result.Context;
  const QualType ValueType = Call->getType();

  // Rvalue optional uses && overload which moves. Suppress if move is cheap.
  if (!ObjExpr->isLValue() &&
      utils::type_traits::hasNonTrivialMoveConstructor(ValueType))
    return;

  const bool IsExpensiveType =
      utils::type_traits::isExpensiveToCopy(ValueType, Ctx).value_or(false);
  const int64_t ValueSize = Ctx.getTypeSizeInChars(ValueType).getQuantity();
  const bool IsExpensive =
      IsExpensiveType || ValueSize > static_cast<int64_t>(SizeThreshold);

  if (!IsExpensive)
    return;

  const CXXMethodDecl *Method = Call->getMethodDecl();
  const CXXRecordDecl *OptionalClass = Method->getParent();
  const Expr *FallbackArg = Call->getArg(0)->IgnoreImplicit();
  const bool HasSideEffects = FallbackArg->HasSideEffects(Ctx);

  {
    auto Diag = diag(Call->getExprLoc(), "'%0' copies expensive type %1; %2")
                << Method->getName() << ValueType
                << buildSuggestion(OptionalClass);

    if (!HasSideEffects) {
      if (auto Fix = buildFixIt(Call, ObjExpr, FallbackArg, OptionalClass,
                                Ctx.getSourceManager(), Ctx.getLangOpts()))
        Diag << *Fix;
    }
  }

  if (HasSideEffects)
    diag(FallbackArg->getExprLoc(),
         "the fallback is always evaluated; a conditional rewrite would "
         "change evaluation semantics",
         DiagnosticIDs::Note);
}

} // namespace clang::tidy::performance
