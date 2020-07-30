//===--- SpuriouslyWakeUpFunctionsCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpuriouslyWakeUpFunctionsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

void SpuriouslyWakeUpFunctionsCheck::registerMatchers(MatchFinder *Finder) {

  auto hasUniqueLock = hasDescendant(declRefExpr(
      hasDeclaration(varDecl(hasType(recordDecl(classTemplateSpecializationDecl(
          hasName("::std::unique_lock"),
          hasTemplateArgument(
              0, templateArgument(refersToType(qualType(hasDeclaration(
                     cxxRecordDecl(hasName("::std::mutex"))))))))))))));

  auto hasWaitDescendantCPP = hasDescendant(
      cxxMemberCallExpr(
          anyOf(
              allOf(hasDescendant(memberExpr(hasDeclaration(functionDecl(
                        allOf(hasName("::std::condition_variable::wait"),
                              parameterCountIs(1)))))),
                    onImplicitObjectArgument(
                        declRefExpr(to(varDecl(hasType(references(recordDecl(
                            hasName("::std::condition_variable")))))))),
                    hasUniqueLock),
              allOf(hasDescendant(memberExpr(hasDeclaration(functionDecl(
                        allOf(hasName("::std::condition_variable::wait_for"),
                              parameterCountIs(2)))))),
                    onImplicitObjectArgument(
                        declRefExpr(to(varDecl(hasType(references(recordDecl(
                            hasName("::std::condition_variable")))))))),
                    hasUniqueLock),
              allOf(hasDescendant(memberExpr(hasDeclaration(functionDecl(
                        allOf(hasName("::std::condition_variable::wait_until"),
                              parameterCountIs(2)))))),
                    onImplicitObjectArgument(
                        declRefExpr(to(varDecl(hasType(references(recordDecl(
                            hasName("::std::condition_variable")))))))),
                    hasUniqueLock)

                  ))
          .bind("wait"));

  auto hasWaitDescendantC = hasDescendant(
      callExpr(callee(functionDecl(hasAnyName("cnd_wait", "cnd_timedwait"))))
          .bind("wait"));
  if (getLangOpts().CPlusPlus) {
    // Check for `CON54-CPP`
    Finder->addMatcher(
        ifStmt(

            allOf(hasWaitDescendantCPP,
                  unless(anyOf(hasDescendant(ifStmt(hasWaitDescendantCPP)),
                               hasDescendant(whileStmt(hasWaitDescendantCPP)),
                               hasDescendant(forStmt(hasWaitDescendantCPP)),
                               hasDescendant(doStmt(hasWaitDescendantCPP)))))

                ),
        this);
  } else {
    // Check for `CON36-C`
    Finder->addMatcher(

        ifStmt(
            allOf(hasWaitDescendantC,
                  unless(anyOf(hasDescendant(ifStmt(hasWaitDescendantC)),
                               hasDescendant(whileStmt(hasWaitDescendantC)),
                               hasDescendant(forStmt(hasWaitDescendantC)),
                               hasDescendant(doStmt(hasWaitDescendantC)),
                               hasParent(whileStmt()),
                               hasParent(compoundStmt(hasParent(whileStmt()))),
                               hasParent(forStmt()),
                               hasParent(compoundStmt(hasParent(forStmt()))),
                               hasParent(doStmt()),
                               hasParent(compoundStmt(hasParent(doStmt())))))

                      ))

            ,
        this);
  }
}

void SpuriouslyWakeUpFunctionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedWait = Result.Nodes.getNodeAs<CallExpr>("wait");
  StringRef WaitName = MatchedWait->getDirectCallee()->getName();
  diag(MatchedWait->getExprLoc(),
       "'%0' should be placed inside a while statement %select{|or used with a "
       "conditional parameter}1")
      << WaitName << (WaitName != "cnd_wait" && WaitName != "cnd_timedwait");
}
} // namespace bugprone
} // namespace tidy
} // namespace clang
