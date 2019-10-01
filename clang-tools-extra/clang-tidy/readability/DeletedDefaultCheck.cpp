//===--- DeletedDefaultCheck.cpp - clang-tidy------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeletedDefaultCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

void DeletedDefaultCheck::registerMatchers(MatchFinder *Finder) {
  // We match constructors/assignment operators that are:
  //   - explicitly marked '= default'
  //   - actually deleted
  //   - not in template instantiation.
  // We bind the declaration to "method-decl" and also to "constructor" when
  // it is a constructor.

  Finder->addMatcher(
      cxxMethodDecl(anyOf(cxxConstructorDecl().bind("constructor"),
                          isCopyAssignmentOperator(),
                          isMoveAssignmentOperator()),
                    isDefaulted(), unless(isImplicit()), isDeleted(),
                    unless(isInstantiated()))
          .bind("method-decl"),
      this);
}

void DeletedDefaultCheck::check(const MatchFinder::MatchResult &Result) {
  const StringRef Message = "%0 is explicitly defaulted but implicitly "
                            "deleted, probably because %1; definition can "
                            "either be removed or explicitly deleted";
  if (const auto *Constructor =
          Result.Nodes.getNodeAs<CXXConstructorDecl>("constructor")) {
    auto Diag = diag(Constructor->getBeginLoc(), Message);
    if (Constructor->isDefaultConstructor()) {
      Diag << "default constructor"
           << "a non-static data member or a base class is lacking a default "
              "constructor";
    } else if (Constructor->isCopyConstructor()) {
      Diag << "copy constructor"
           << "a non-static data member or a base class is not copyable";
    } else if (Constructor->isMoveConstructor()) {
      Diag << "move constructor"
           << "a non-static data member or a base class is neither copyable "
              "nor movable";
    }
  } else if (const auto *Assignment =
                 Result.Nodes.getNodeAs<CXXMethodDecl>("method-decl")) {
    diag(Assignment->getBeginLoc(), Message)
        << (Assignment->isCopyAssignmentOperator() ? "copy assignment operator"
                                                   : "move assignment operator")
        << "a base class or a non-static data member is not assignable, e.g. "
           "because the latter is marked 'const'";
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
