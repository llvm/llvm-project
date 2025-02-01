//===--- SmartptrResetAmbiguousCallCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartptrResetAmbiguousCallCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

AST_MATCHER_P(CallExpr, everyArgumentMatches,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  for (const auto *Arg : Node.arguments()) {
    if (!InnerMatcher.matches(*Arg, Finder, Builder))
      return false;
  }

  return true;
}

AST_MATCHER(CXXMethodDecl, hasOnlyDefaultParameters) {
  for (const auto *Param : Node.parameters()) {
    if (!Param->hasDefaultArg())
      return false;
  }

  return true;
}

const auto DefaultSmartPointers = "::std::shared_ptr;::std::unique_ptr";
} // namespace

SmartptrResetAmbiguousCallCheck::SmartptrResetAmbiguousCallCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SmartPointers(utils::options::parseStringList(
          Options.get("SmartPointers", DefaultSmartPointers))) {}

void SmartptrResetAmbiguousCallCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SmartPointers",
                utils::options::serializeStringList(SmartPointers));
}

void SmartptrResetAmbiguousCallCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsSmartptr = hasAnyName(SmartPointers);

  const auto ResetMethod =
      cxxMethodDecl(hasName("reset"), hasOnlyDefaultParameters());

  const auto TypeWithReset =
      anyOf(cxxRecordDecl(hasMethod(ResetMethod)),
            classTemplateSpecializationDecl(
                hasSpecializedTemplate(classTemplateDecl(has(ResetMethod)))));

  const auto SmartptrWithBugproneReset = classTemplateSpecializationDecl(
      IsSmartptr,
      hasTemplateArgument(
          0, templateArgument(refersToType(hasUnqualifiedDesugaredType(
                 recordType(hasDeclaration(TypeWithReset)))))));

  // Find a.reset() calls
  Finder->addMatcher(
      cxxMemberCallExpr(callee(memberExpr(member(hasName("reset")))),
                        everyArgumentMatches(cxxDefaultArgExpr()),
                        on(expr(hasType(SmartptrWithBugproneReset))))
          .bind("smartptrResetCall"),
      this);

  // Find a->reset() calls
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(memberExpr(
              member(ResetMethod),
              hasObjectExpression(
                  cxxOperatorCallExpr(
                      hasOverloadedOperatorName("->"),
                      hasArgument(
                          0, expr(hasType(
                                 classTemplateSpecializationDecl(IsSmartptr)))))
                      .bind("OpCall")))),
          everyArgumentMatches(cxxDefaultArgExpr()))
          .bind("objectResetCall"),
      this);
}

void SmartptrResetAmbiguousCallCheck::check(
    const MatchFinder::MatchResult &Result) {

  if (const auto *SmartptrResetCall =
          Result.Nodes.getNodeAs<CXXMemberCallExpr>("smartptrResetCall")) {
    const auto *Member = cast<MemberExpr>(SmartptrResetCall->getCallee());

    diag(SmartptrResetCall->getBeginLoc(),
         "be explicit when calling 'reset()' on a smart pointer with a "
         "pointee that has a 'reset()' method");

    diag(SmartptrResetCall->getBeginLoc(), "assign the pointer to 'nullptr'",
         DiagnosticIDs::Note)
        << FixItHint::CreateReplacement(
               SourceRange(Member->getOperatorLoc(),
                           Member->getOperatorLoc().getLocWithOffset(0)),
               " =")
        << FixItHint::CreateReplacement(
               SourceRange(Member->getMemberLoc(),
                           SmartptrResetCall->getEndLoc()),
               " nullptr");
    return;
  }

  if (const auto *ObjectResetCall =
          Result.Nodes.getNodeAs<CXXMemberCallExpr>("objectResetCall")) {
    const auto *Arrow = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("OpCall");

    const CharSourceRange SmartptrSourceRange =
        Lexer::getAsCharRange(Arrow->getArg(0)->getSourceRange(),
                              *Result.SourceManager, getLangOpts());

    diag(ObjectResetCall->getBeginLoc(),
         "be explicit when calling 'reset()' on a pointee of a smart pointer");

    diag(ObjectResetCall->getBeginLoc(),
         "use dereference to call 'reset' method of the pointee",
         DiagnosticIDs::Note)
        << FixItHint::CreateInsertion(SmartptrSourceRange.getBegin(), "(*")
        << FixItHint::CreateInsertion(SmartptrSourceRange.getEnd(), ")")
        << FixItHint::CreateReplacement(
               CharSourceRange::getCharRange(
                   Arrow->getOperatorLoc(),
                   Arrow->getOperatorLoc().getLocWithOffset(2)),
               ".");
  }
}

} // namespace clang::tidy::bugprone
