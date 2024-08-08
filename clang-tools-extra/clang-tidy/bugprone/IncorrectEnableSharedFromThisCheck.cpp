//===--- IncorrectEnableSharedFromThisCheck.cpp - clang-tidy
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectEnableSharedFromThisCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void IncorrectEnableSharedFromThisCheck::registerMatchers(
    MatchFinder *match_finder) {
  match_finder->addMatcher(
      cxxRecordDecl(
          unless(isExpansionInSystemHeader()),
          hasAnyBase(cxxBaseSpecifier(unless(isPublic()),
                                      hasType(cxxRecordDecl(hasName(
                                          "::std::enable_shared_from_this"))))))
          .bind("class-base-std-enable"),
      this);
  match_finder->addMatcher(
      cxxRecordDecl(unless(isExpansionInSystemHeader()),
                    hasAnyBase(cxxBaseSpecifier(hasType(
                        cxxRecordDecl(hasName("enable_shared_from_this"))))))
          .bind("class-missing-std"),
      this);
}

void IncorrectEnableSharedFromThisCheck::check(
    const MatchFinder::MatchResult &result) {
  const auto *StdEnableSharedClassDecl =
      result.Nodes.getNodeAs<CXXRecordDecl>("class-base-std-enable");
  const auto *MissingStdSharedClassDecl =
      result.Nodes.getNodeAs<CXXRecordDecl>("class-missing-std");

  if (StdEnableSharedClassDecl) {
    for (const auto &Base : StdEnableSharedClassDecl->bases()) {
      const auto *BaseType = Base.getType()->getAsCXXRecordDecl();
      if (BaseType && BaseType->getQualifiedNameAsString() ==
                          "std::enable_shared_from_this") {
        SourceRange ReplacementRange = Base.getSourceRange();
        std::string ReplacementString =
            "public " + Base.getType().getAsString();
        FixItHint Hint =
            FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
        diag(
            StdEnableSharedClassDecl->getLocation(),
            "inheritance from std::enable_shared_from_this should be public "
            "inheritance, otherwise the internal weak_ptr won't be initialized",
            DiagnosticIDs::Warning)
            << Hint;
        break;
      }
    }
  }

  if (MissingStdSharedClassDecl) {
    for (const auto &Base : MissingStdSharedClassDecl->bases()) {
      const auto *BaseType = Base.getType()->getAsCXXRecordDecl();
      if (BaseType &&
          BaseType->getQualifiedNameAsString() == "enable_shared_from_this") {
        clang::AccessSpecifier AccessSpec = Base.getAccessSpecifier();
        if (AccessSpec == clang::AS_public) {
          SourceLocation InsertLocation = Base.getBaseTypeLoc();
          FixItHint Hint = FixItHint::CreateInsertion(InsertLocation, "std::");
          diag(MissingStdSharedClassDecl->getLocation(),
               "Should be std::enable_shared_from_this", DiagnosticIDs::Warning)
              << Hint;
          break;
        } else {
          SourceRange ReplacementRange = Base.getSourceRange();
          std::string ReplacementString =
              "public std::" + Base.getType().getAsString();
          FixItHint Hint =
              FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
          diag(MissingStdSharedClassDecl->getLocation(),
               "Should be std::enable_shared_from_this and "
               "inheritance from std::enable_shared_from_this should be public "
               "inheritance, otherwise the internal weak_ptr won't be "
               "initialized",
               DiagnosticIDs::Warning)
              << Hint;
          break;
        }
      }
    }
  }
}
} // namespace clang::tidy::bugprone
