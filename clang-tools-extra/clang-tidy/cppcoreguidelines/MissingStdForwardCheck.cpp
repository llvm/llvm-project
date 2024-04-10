//===--- MissingStdForwardCheck.cpp - clang-tidy --------------------------===//
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
  ast_matchers::internal::Matcher<QualType> Inner = possiblyPackExpansionOf(
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

  QualType ParamType =
      Node.getType().getNonPackExpansionType()->getPointeeType();
  const auto *TemplateType = ParamType->getAs<TemplateTypeParmType>();
  if (!TemplateType)
    return false;

  return TemplateType->getDepth() ==
         FuncTemplate->getTemplateParameters()->getDepth();
}

AST_MATCHER_P(NamedDecl, hasSameNameAsBoundNode, std::string, BindingID) {
  IdentifierInfo *II = Node.getIdentifier();
  if (nullptr == II)
    return false;
  StringRef Name = II->getName();

  return Builder->removeBindings(
      [this, Name](const ast_matchers::internal::BoundNodesMap &Nodes) {
        const DynTypedNode &BN = Nodes.getNode(this->BindingID);
        if (const auto *ND = BN.get<NamedDecl>()) {
          if (!isa<FieldDecl, CXXMethodDecl, VarDecl>(ND))
            return true;
          return ND->getName() != Name;
        }
        return true;
      });
}

AST_MATCHER_P(LambdaCapture, hasCaptureKind, LambdaCaptureKind, Kind) {
  return Node.getCaptureKind() == Kind;
}

AST_MATCHER_P(LambdaExpr, hasCaptureDefaultKind, LambdaCaptureDefault, Kind) {
  return Node.getCaptureDefault() == Kind;
}

AST_MATCHER(VarDecl, hasIdentifier) {
  const IdentifierInfo *ID = Node.getIdentifier();
  return ID != NULL && !ID->isPlaceholder();
}

} // namespace

void MissingStdForwardCheck::registerMatchers(MatchFinder *Finder) {
  auto RefToParmImplicit = allOf(
      equalsBoundNode("var"), hasInitializer(ignoringParenImpCasts(
                                  declRefExpr(to(equalsBoundNode("param"))))));
  auto RefToParm = capturesVar(
      varDecl(anyOf(hasSameNameAsBoundNode("param"), RefToParmImplicit)));
  auto HasRefToParm = hasAnyCapture(RefToParm);

  auto CaptureInRef =
      allOf(hasCaptureDefaultKind(LambdaCaptureDefault::LCD_ByRef),
            unless(hasAnyCapture(
                capturesVar(varDecl(hasSameNameAsBoundNode("param"))))));
  auto CaptureInCopy = allOf(
      hasCaptureDefaultKind(LambdaCaptureDefault::LCD_ByCopy), HasRefToParm);
  auto CaptureByRefExplicit = hasAnyCapture(
      allOf(hasCaptureKind(LambdaCaptureKind::LCK_ByRef), RefToParm));

  auto CapturedInBody =
      lambdaExpr(anyOf(CaptureInRef, CaptureInCopy, CaptureByRefExplicit));
  auto CapturedInCaptureList = hasAnyCapture(capturesVar(
      varDecl(hasInitializer(ignoringParenImpCasts(equalsBoundNode("call"))))));

  auto CapturedInLambda = hasDeclContext(cxxRecordDecl(
      isLambda(),
      hasParent(lambdaExpr(forCallable(equalsBoundNode("func")),
                           anyOf(CapturedInCaptureList, CapturedInBody)))));

  auto ToParam = hasAnyParameter(parmVarDecl(equalsBoundNode("param")));

  auto ForwardCallMatcher = callExpr(
      callExpr().bind("call"), argumentCountIs(1),
      hasArgument(0, declRefExpr(to(varDecl().bind("var")))),
      forCallable(
          anyOf(allOf(equalsBoundNode("func"),
                      functionDecl(hasAnyParameter(parmVarDecl(allOf(
                          equalsBoundNode("param"), equalsBoundNode("var")))))),
                CapturedInLambda)),
      callee(unresolvedLookupExpr(hasAnyDeclaration(
          namedDecl(hasUnderlyingDecl(hasName("::std::forward")))))),

      unless(anyOf(hasAncestor(typeLoc()),
                   hasAncestor(expr(hasUnevaluatedContext())))));

  Finder->addMatcher(
      parmVarDecl(
          parmVarDecl().bind("param"), hasIdentifier(),
          unless(hasAttr(attr::Kind::Unused)), isTemplateTypeParameter(),
          hasAncestor(functionDecl().bind("func")),
          hasAncestor(functionDecl(
              isDefinition(), equalsBoundNode("func"), ToParam,
              unless(anyOf(isDeleted(),
                           hasDescendant(std::move(ForwardCallMatcher))))))),
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

} // namespace clang::tidy::cppcoreguidelines
