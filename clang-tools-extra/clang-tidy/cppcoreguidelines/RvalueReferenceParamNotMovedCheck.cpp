//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RvalueReferenceParamNotMovedCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

using matchers::hasUnevaluatedContext;

namespace {
AST_MATCHER_P(LambdaExpr, valueCapturesVar, DeclarationMatcher, VarMatcher) {
  return std::find_if(Node.capture_begin(), Node.capture_end(),
                      [&](const LambdaCapture &Capture) {
                        return Capture.capturesVariable() &&
                               VarMatcher.matches(*Capture.getCapturedVar(),
                                                  Finder, Builder) &&
                               Capture.getCaptureKind() == LCK_ByCopy;
                      }) != Node.capture_end();
}
AST_MATCHER_P2(Stmt, argumentOf, bool, AllowPartialMove, StatementMatcher,
               Ref) {
  if (AllowPartialMove)
    return stmt(anyOf(Ref, hasDescendant(Ref))).matches(Node, Finder, Builder);
  return Ref.matches(Node, Finder, Builder);
}
} // namespace

void RvalueReferenceParamNotMovedCheck::registerMatchers(MatchFinder *Finder) {
  auto ToParam = hasAnyParameter(parmVarDecl(equalsBoundNode("param")));

  const StatementMatcher MoveCallMatcher =
      callExpr(
          argumentCountIs(1),
          anyOf(callee(functionDecl(hasName(MoveFunction))),
                callee(unresolvedLookupExpr(hasAnyDeclaration(
                    namedDecl(hasUnderlyingDecl(hasName(MoveFunction))))))),
          hasArgument(
              0, argumentOf(
                     AllowPartialMove,
                     declRefExpr(to(equalsBoundNode("param"))).bind("ref"))),
          unless(hasAncestor(
              lambdaExpr(valueCapturesVar(equalsBoundNode("param"))))),
          unless(anyOf(hasAncestor(typeLoc()),
                       hasAncestor(expr(hasUnevaluatedContext())))))
          .bind("move-call");

  // P1825R0 (C++20): returning a named rvalue reference parameter by name
  // performs an implicit move, which is equivalent to ``std::move(param)``
  const StatementMatcher ImplicitMoveReturnMatcher = traverse(
      TK_IgnoreUnlessSpelledInSource,
      returnStmt(hasReturnValue(ignoringParens(
                     declRefExpr(to(equalsBoundNode("param"))).bind("ref"))))
          .bind("implicit-move-return"));

  const bool EnableImplicitMove =
      AllowImplicitMove && getLangOpts().CPlusPlus20;

  const StatementMatcher UsageMatcher = stmt(
      anyOf(MoveCallMatcher, EnableImplicitMove ? ImplicitMoveReturnMatcher
                                                : stmt(unless(anything()))));

  Finder->addMatcher(
      parmVarDecl(
          hasType(type(rValueReferenceType())), parmVarDecl().bind("param"),
          unless(hasType(references(qualType(
              anyOf(isConstQualified(), substTemplateTypeParmType()))))),
          optionally(hasType(qualType(references(templateTypeParmType(
              hasDeclaration(templateTypeParmDecl().bind("template-type"))))))),
          hasDeclContext(
              functionDecl(
                  isDefinition(), unless(isDeleted()), unless(isDefaulted()),
                  unless(isImplicit()),
                  unless(cxxConstructorDecl(isMoveConstructor())),
                  unless(cxxMethodDecl(isMoveAssignmentOperator())), ToParam,
                  anyOf(cxxConstructorDecl(
                            optionally(hasDescendant(UsageMatcher))),
                        functionDecl(
                            unless(cxxConstructorDecl()),
                            optionally(hasBody(hasDescendant(UsageMatcher))))))
                  .bind("func"))),
      this);
}

void RvalueReferenceParamNotMovedCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *TemplateType =
      Result.Nodes.getNodeAs<TemplateTypeParmDecl>("template-type");

  if (!Param || !Function)
    return;

  if (IgnoreUnnamedParams && Param->getName().empty())
    return;

  if (!Param->isUsed() && Param->hasAttr<UnusedAttr>())
    return;

  if (IgnoreNonDeducedTemplateTypes && TemplateType)
    return;

  if (TemplateType) {
    if (const FunctionTemplateDecl *FuncTemplate =
            Function->getDescribedFunctionTemplate()) {
      const TemplateParameterList *Params =
          FuncTemplate->getTemplateParameters();
      if (llvm::is_contained(*Params, TemplateType)) {
        // Ignore forwarding reference
        return;
      }
    }
  }

  const auto *MoveCall = Result.Nodes.getNodeAs<CallExpr>("move-call");
  const auto *ImplicitMoveReturn =
      Result.Nodes.getNodeAs<ReturnStmt>("implicit-move-return");
  if (MoveCall || ImplicitMoveReturn)
    return;

  diag(Param->getLocation(),
       "rvalue reference parameter %0 is never moved from "
       "inside the function body")
      << Param;
}

RvalueReferenceParamNotMovedCheck::RvalueReferenceParamNotMovedCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowPartialMove(Options.get("AllowPartialMove", false)),
      IgnoreUnnamedParams(Options.get("IgnoreUnnamedParams", false)),
      IgnoreNonDeducedTemplateTypes(
          Options.get("IgnoreNonDeducedTemplateTypes", false)),
      AllowImplicitMove(Options.get("AllowImplicitMove", false)),
      MoveFunction(Options.get("MoveFunction", "::std::move")) {}

void RvalueReferenceParamNotMovedCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowPartialMove", AllowPartialMove);
  Options.store(Opts, "IgnoreUnnamedParams", IgnoreUnnamedParams);
  Options.store(Opts, "IgnoreNonDeducedTemplateTypes",
                IgnoreNonDeducedTemplateTypes);
  Options.store(Opts, "AllowImplicitMove", AllowImplicitMove);
  Options.store(Opts, "MoveFunction", MoveFunction);
}

} // namespace clang::tidy::cppcoreguidelines
