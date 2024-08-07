//===--- PublicEnableSharedFromThisCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PublicEnableSharedFromThisCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

  void PublicEnableSharedFromThisCheck::registerMatchers(MatchFinder *match_finder) {
      match_finder->addMatcher(
              cxxRecordDecl(
                  hasAnyBase(
                      cxxBaseSpecifier(unless(isPublic()), 
                      hasType(
                          cxxRecordDecl(
                              hasName("::std::enable_shared_from_this"))))
                      )
                  )
              .bind("not-public-enable-shared"), this);
  }

  void PublicEnableSharedFromThisCheck::check(const MatchFinder::MatchResult &result) {
      const auto *EnableSharedClassDecl =
          result.Nodes.getNodeAs<CXXRecordDecl>("not-public-enable-shared");

      for (const auto &Base : EnableSharedClassDecl->bases()) {
          const auto *BaseType = Base.getType()->getAsCXXRecordDecl();
          if (BaseType && BaseType->getQualifiedNameAsString() == "std::enable_shared_from_this") {
              SourceLocation InsertLoc = Base.getBeginLoc();
              FixItHint Hint = FixItHint::CreateInsertion(InsertLoc, "public ");
              diag(EnableSharedClassDecl->getLocation(), "class %0 is not public even though it's derived from std::enable_shared_from_this", DiagnosticIDs::Warning)
              << EnableSharedClassDecl->getNameAsString()
              << Hint;
              break;
          }
      }
  }
} // namespace clang::tidy::bugprone
