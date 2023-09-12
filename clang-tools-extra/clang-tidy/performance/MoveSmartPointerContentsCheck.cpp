//===--- MoveSmartPointerContentsCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "MoveSmartPointerContentsCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

MoveSmartPointerContentsCheck::MoveSmartPointerContentsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UniquePointerClasses(utils::options::parseStringList(
          Options.get("UniquePointerClasses", "std::unique_ptr"))),
      IsAUniquePointer(namedDecl(hasAnyName(UniquePointerClasses))),
      SharedPointerClasses(utils::options::parseStringList(
          Options.get("SharedPointerClasses", "std::shared_ptr"))),
      IsASharedPointer(namedDecl(hasAnyName(SharedPointerClasses))) {}

void MoveSmartPointerContentsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UniquePtrClasses",
                utils::options::serializeStringList(UniquePointerClasses));
  Options.store(Opts, "SharedPtrClasses",
                utils::options::serializeStringList(SharedPointerClasses));
}

void MoveSmartPointerContentsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("std::move"))),
          hasArgument(0, cxxOperatorCallExpr(hasOperatorName("*"),
                                             hasDeclaration(cxxMethodDecl(ofClass(IsAUniquePointer)))).bind("unique_op")))   
          .bind("unique_call"),
      this);

  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("std::move"))),
          hasArgument(0, cxxOperatorCallExpr(hasOperatorName("*"),
                                             hasDeclaration(cxxMethodDecl(ofClass(IsASharedPointer)))).bind("shared_op")))   
          .bind("shared_call"),
      this);
}
  
void MoveSmartPointerContentsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *UniqueCall = Result.Nodes.getNodeAs<CallExpr>("unique_call");
  const auto *SharedCall = Result.Nodes.getNodeAs<CallExpr>("shared_call");

  if (UniqueCall) {
    const auto* UniqueOp = Result.Nodes.getNodeAs<Expr>("unique_op");

    diag(UniqueCall->getBeginLoc(),
         "prefer to move the smart pointer rather than its contents") << FixItHint::CreateInsertion(UniqueCall->getBeginLoc(), "*")
								      << FixItHint::CreateRemoval(UniqueOp->getBeginLoc());
  }
  if (SharedCall) {
    const auto* SharedOp = Result.Nodes.getNodeAs<Expr>("shared_op");

    diag(SharedCall->getBeginLoc(),
         "don't move the contents out of a shared pointer, as other accessors "
         "expect them to remain in a determinate state") << FixItHint::CreateInsertion(SharedCall->getBeginLoc(), "*")
							 << FixItHint::CreateRemoval(SharedOp->getBeginLoc());
  }
}

} // namespace clang::tidy::performance
