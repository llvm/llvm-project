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
#include "clang/Tooling/FixIt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {

AST_MATCHER(Expr, isLValueExpr) { return Node.isLValue(); }

AST_MATCHER(QualType, hasNonTrivialMoveCtor) {
  return utils::type_traits::hasNonTrivialMoveConstructor(Node);
}

AST_MATCHER_P(QualType, isLargerThan, unsigned, SizeThreshold) {
  if (Node.isNull() || Node->isDependentType() || Node->isIncompleteType())
    return false;
  return Finder->getASTContext().getTypeSizeInChars(Node).getQuantity() >
         static_cast<int64_t>(SizeThreshold);
}

} // namespace

static bool hasOperatorStar(const CXXRecordDecl *RD) {
  const DeclarationName OpStar =
      RD->getASTContext().DeclarationNames.getCXXOperatorName(OO_Star);
  return !RD->lookup(OpStar).empty();
}

static StringRef findValueMethod(const CXXRecordDecl *RD) {
  ASTContext &Ctx = RD->getASTContext();
  for (StringRef Name : {"value", "Value"}) {
    const DeclarationName DN = &Ctx.Idents.get(Name);
    if (!RD->lookup(DN).empty())
      return Name;
  }
  return {};
}

static std::string buildSuggestion(const CXXRecordDecl *OptionalClass) {
  const bool HasDeref = hasOperatorStar(OptionalClass);
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

static std::optional<FixItHint> buildFixIt(const CXXMemberCallExpr *Call,
                                           const Expr *ObjExpr,
                                           const Expr *FallbackArg,
                                           const CXXRecordDecl *OptionalClass,
                                           const ASTContext &Ctx) {
  if (Call->getBeginLoc().isMacroID())
    return std::nullopt;
  if (!ObjExpr->isLValue())
    return std::nullopt;
  if (ObjExpr->HasSideEffects(Ctx))
    return std::nullopt;
  if (!hasOperatorStar(OptionalClass))
    return std::nullopt;

  StringRef ObjText = tooling::fixit::getText(*ObjExpr, Ctx);
  StringRef ArgText = tooling::fixit::getText(*FallbackArg, Ctx);

  if (ObjText.empty() || ArgText.empty())
    return std::nullopt;

  const std::string Replacement =
      ("(" + ObjText + " ? *" + ObjText + " : " + ArgText + ")").str();
  return tooling::fixit::createReplacement(*Call, Replacement);
}

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
      callee(cxxMethodDecl(ValueOrMatcher, ofClass(OptionalTypesMatcher))),
      anyOf(on(isLValueExpr()),
            hasType(qualType(unless(hasNonTrivialMoveCtor())))),
      hasType(qualType(
          anyOf(matchers::isExpensiveToCopy(), isLargerThan(SizeThreshold)))),
      unless(anyOf(hasAncestor(typeLoc()),
                   hasAncestor(expr(matchers::hasUnevaluatedContext())))));

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

  const CXXMethodDecl *Method = Call->getMethodDecl();
  const CXXRecordDecl *OptionalClass = Method->getParent();
  const Expr *FallbackArg = Call->getArg(0)->IgnoreImplicit();
  const bool HasSideEffects = FallbackArg->HasSideEffects(Ctx);

  {
    auto Diag = diag(Call->getExprLoc(), "'%0' copies expensive type %1; %2")
                << Method->getName() << ValueType
                << buildSuggestion(OptionalClass);

    if (!HasSideEffects) {
      if (auto Fix = buildFixIt(Call, ObjExpr, FallbackArg, OptionalClass, Ctx))
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
