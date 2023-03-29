//===--- RvalueReferenceParamNotMovedCheck.cpp - clang-tidy ---------------===//
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
  if (AllowPartialMove) {
    return stmt(anyOf(Ref, hasDescendant(Ref))).matches(Node, Finder, Builder);
  } else {
    return Ref.matches(Node, Finder, Builder);
  }
}
} // namespace

void RvalueReferenceParamNotMovedCheck::registerMatchers(MatchFinder *Finder) {
  auto ToParam = hasAnyParameter(parmVarDecl(equalsBoundNode("param")));

  StatementMatcher MoveCallMatcher =
      callExpr(
          argumentCountIs(1),
          anyOf(callee(functionDecl(hasName("::std::move"))),
                callee(unresolvedLookupExpr(hasAnyDeclaration(
                    namedDecl(hasUnderlyingDecl(hasName("::std::move"))))))),
          hasArgument(
              0, argumentOf(
                     AllowPartialMove,
                     declRefExpr(to(equalsBoundNode("param"))).bind("ref"))),
          unless(hasAncestor(
              lambdaExpr(valueCapturesVar(equalsBoundNode("param"))))),
          unless(anyOf(hasAncestor(typeLoc()),
                       hasAncestor(expr(hasUnevaluatedContext())))))
          .bind("move-call");

  Finder->addMatcher(
      parmVarDecl(
          hasType(type(rValueReferenceType())), parmVarDecl().bind("param"),
          unless(hasType(references(qualType(
              anyOf(isConstQualified(), substTemplateTypeParmType()))))),
          optionally(hasType(qualType(references(templateTypeParmType(
              hasDeclaration(templateTypeParmDecl().bind("template-type"))))))),
          anyOf(hasAncestor(cxxConstructorDecl(
                    ToParam, isDefinition(), unless(isMoveConstructor()),
                    optionally(hasDescendant(MoveCallMatcher)))),
                hasAncestor(functionDecl(
                    unless(cxxConstructorDecl()), ToParam,
                    unless(cxxMethodDecl(isMoveAssignmentOperator())),
                    hasBody(optionally(hasDescendant(MoveCallMatcher))))))),
      this);
}

void RvalueReferenceParamNotMovedCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *TemplateType =
      Result.Nodes.getNodeAs<TemplateTypeParmDecl>("template-type");

  if (!Param)
    return;

  if (IgnoreUnnamedParams && Param->getName().empty())
    return;

  const auto *Function = dyn_cast<FunctionDecl>(Param->getDeclContext());
  if (!Function)
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
  if (!MoveCall) {
    diag(Param->getLocation(),
         "rvalue reference parameter %0 is never moved from "
         "inside the function body")
        << Param;
  }
}

RvalueReferenceParamNotMovedCheck::RvalueReferenceParamNotMovedCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowPartialMove(Options.getLocalOrGlobal("AllowPartialMove", false)),
      IgnoreUnnamedParams(
          Options.getLocalOrGlobal("IgnoreUnnamedParams", false)),
      IgnoreNonDeducedTemplateTypes(
          Options.getLocalOrGlobal("IgnoreNonDeducedTemplateTypes", false)) {}

void RvalueReferenceParamNotMovedCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowPartialMove", AllowPartialMove);
  Options.store(Opts, "IgnoreUnnamedParams", IgnoreUnnamedParams);
  Options.store(Opts, "IgnoreNonDeducedTemplateTypes",
                IgnoreNonDeducedTemplateTypes);
}

} // namespace clang::tidy::cppcoreguidelines
