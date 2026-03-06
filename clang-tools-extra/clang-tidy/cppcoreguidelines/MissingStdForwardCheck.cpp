//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MissingStdForwardCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/IdentifierTable.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {

using matchers::hasUnevaluatedContext;

AST_MATCHER_P(QualType, possiblyPackExpansionOf,
              ast_matchers::internal::Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getNonPackExpansionType(), Finder, Builder);
}

AST_MATCHER(ParmVarDecl, isTemplateTypeParameter) {
  const ast_matchers::internal::Matcher<QualType> Inner =
      possiblyPackExpansionOf(
          qualType(rValueReferenceType(),
                   references(templateTypeParmType(
                       hasDeclaration(templateTypeParmDecl()))),
                   unless(references(qualType(isConstQualified())))));
  if (!Inner.matches(Node.getType(), Finder, Builder))
    return false;

  const auto *Function = dyn_cast<FunctionDecl>(Node.getDeclContext());
  if (!Function)
    return false;

  const FunctionTemplateDecl *FuncTemplate =
      Function->getDescribedFunctionTemplate();
  if (!FuncTemplate)
    return false;

  const QualType ParamType =
      Node.getType().getNonPackExpansionType()->getPointeeType();
  const auto *TemplateType = ParamType->getAsCanonical<TemplateTypeParmType>();
  if (!TemplateType)
    return false;

  return TemplateType->getDepth() ==
         FuncTemplate->getTemplateParameters()->getDepth();
}

AST_MATCHER_P(LambdaCapture, hasCaptureKind, LambdaCaptureKind, Kind) {
  return Node.getCaptureKind() == Kind;
}

AST_MATCHER_P(LambdaExpr, hasCaptureDefaultKind, LambdaCaptureDefault, Kind) {
  return Node.getCaptureDefault() == Kind;
}

AST_MATCHER(VarDecl, hasIdentifier) {
  const IdentifierInfo *ID = Node.getIdentifier();
  return ID != nullptr && !ID->isPlaceholder();
}

AST_MATCHER_P(ValueDecl, refersToBoundParm, std::string, ParamID) {
  return Builder->removeBindings(
      [&](const ast_matchers::internal::BoundNodesMap &Nodes) {
        const auto *Param = Nodes.getNodeAs<ParmVarDecl>(ParamID);
        if (!Param)
          return true;

        for (const ValueDecl *V = &Node; V;) {
          if (V == Param)
            return false;

          const auto *VD = dyn_cast<VarDecl>(V);
          const Expr *Init = (VD && VD->getType()->isReferenceType())
                                 ? VD->getInit()
                                 : nullptr;
          const auto *DRE =
              Init ? dyn_cast<DeclRefExpr>(Init->IgnoreParenImpCasts())
                   : nullptr;
          V = DRE ? DRE->getDecl() : nullptr;
        }
        return true;
      });
}

} // namespace

void MissingStdForwardCheck::registerMatchers(MatchFinder *Finder) {
  auto CapturedVar = varDecl(refersToBoundParm("param"));

  auto CaptureInRef =
      allOf(hasCaptureDefaultKind(LambdaCaptureDefault::LCD_ByRef),
            unless(hasAnyCapture(capturesVar(CapturedVar))));
  auto CaptureByRefExplicit = hasAnyCapture(allOf(
      hasCaptureKind(LambdaCaptureKind::LCK_ByRef), capturesVar(CapturedVar)));

  auto CapturedInBody = lambdaExpr(anyOf(CaptureInRef, CaptureByRefExplicit));
  auto CapturedInCaptureList = hasAnyCapture(capturesVar(
      varDecl(hasInitializer(ignoringParenImpCasts(equalsBoundNode("call"))))));

  auto CapturedInLambda = hasDeclContext(cxxRecordDecl(
      isLambda(), hasParent(lambdaExpr(
                      anyOf(CapturedInCaptureList, CapturedInBody),
                      hasAncestor(functionDecl(equalsBoundNode("func")))))));

  auto ToParam = hasAnyParameter(parmVarDecl(equalsBoundNode("param")));

  auto ForwardCallMatcher =
      callExpr(callExpr().bind("call"), argumentCountIs(1),
               hasArgument(0, declRefExpr(to(CapturedVar)).bind("var")),
               forCallable(anyOf(equalsBoundNode("func"), CapturedInLambda)),
               callee(unresolvedLookupExpr(hasAnyDeclaration(
                   namedDecl(hasUnderlyingDecl(hasName(ForwardFunction)))))),

               unless(anyOf(hasAncestor(typeLoc()),
                            hasAncestor(expr(hasUnevaluatedContext())))));

  Finder->addMatcher(
      parmVarDecl(
          parmVarDecl().bind("param"), hasIdentifier(),
          unless(hasAttr(attr::Kind::Unused)), isTemplateTypeParameter(),
          hasAncestor(functionDecl().bind("func")),
          hasAncestor(functionDecl(
              isDefinition(), equalsBoundNode("func"), ToParam,
              unless(anyOf(isDeleted(), hasDescendant(ForwardCallMatcher)))))),
      this);
}

void MissingStdForwardCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");

  if (!Param)
    return;

  diag(Param->getLocation(),
       "forwarding reference parameter %0 is never forwarded "
       "inside the function body")
      << Param;
}

MissingStdForwardCheck::MissingStdForwardCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ForwardFunction(Options.get("ForwardFunction", "::std::forward")) {}

void MissingStdForwardCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ForwardFunction", ForwardFunction);
}

} // namespace clang::tidy::cppcoreguidelines
