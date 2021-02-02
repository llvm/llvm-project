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

  auto HasUniqueLock = hasDescendant(declRefExpr(
      hasDeclaration(varDecl(hasType(recordDecl(classTemplateSpecializationDecl(
          hasName("::std::unique_lock"),
          hasTemplateArgument(
              0, templateArgument(refersToType(qualType(hasDeclaration(
                     cxxRecordDecl(hasName("::std::mutex"))))))))))))));

  auto HasWaitDescendantCpp = hasDescendant(
      cxxMemberCallExpr(
          anyOf(
              allOf(hasDescendant(memberExpr(hasDeclaration(functionDecl(
                        allOf(hasName("::std::condition_variable::wait"),
                              parameterCountIs(1)))))),
                    onImplicitObjectArgument(
                        declRefExpr(to(varDecl(hasType(references(recordDecl(
                            hasName("::std::condition_variable")))))))),
                    HasUniqueLock),
              allOf(hasDescendant(memberExpr(hasDeclaration(functionDecl(
                        allOf(hasName("::std::condition_variable::wait_for"),
                              parameterCountIs(2)))))),
                    onImplicitObjectArgument(
                        declRefExpr(to(varDecl(hasType(references(recordDecl(
                            hasName("::std::condition_variable")))))))),
                    HasUniqueLock),
              allOf(hasDescendant(memberExpr(hasDeclaration(functionDecl(
                        allOf(hasName("::std::condition_variable::wait_until"),
                              parameterCountIs(2)))))),
                    onImplicitObjectArgument(
                        declRefExpr(to(varDecl(hasType(references(recordDecl(
                            hasName("::std::condition_variable")))))))),
                    HasUniqueLock)

                  ))
          .bind("wait"));

  auto HasWaitDescendantC = hasDescendant(
      callExpr(callee(functionDecl(hasAnyName("cnd_wait", "cnd_timedwait"))))
          .bind("wait"));
  if (getLangOpts().CPlusPlus) {
    // Check for `CON54-CPP`
    Finder->addMatcher(
        ifStmt(
            allOf(HasWaitDescendantCpp,
                  unless(anyOf(hasDescendant(ifStmt(HasWaitDescendantCpp)),
                               hasDescendant(whileStmt(HasWaitDescendantCpp)),
                               hasDescendant(forStmt(HasWaitDescendantCpp)),
                               hasDescendant(doStmt(HasWaitDescendantCpp)))))

                ),
        this);
  } else {
    // Check for `CON36-C`
    Finder->addMatcher(
        ifStmt(
            allOf(HasWaitDescendantC,
                  unless(anyOf(hasDescendant(ifStmt(HasWaitDescendantC)),
                               hasDescendant(whileStmt(HasWaitDescendantC)),
                               hasDescendant(forStmt(HasWaitDescendantC)),
                               hasDescendant(doStmt(HasWaitDescendantC)),
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
