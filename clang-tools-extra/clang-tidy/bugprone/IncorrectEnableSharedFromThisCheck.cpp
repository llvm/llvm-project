//===--- IncorrectEnableSharedFromThisCheck.cpp - clang-tidy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectEnableSharedFromThisCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void IncorrectEnableSharedFromThisCheck::registerMatchers(MatchFinder *Finder) {
  const auto EnableSharedFromThis =
      cxxRecordDecl(hasName("enable_shared_from_this"), isInStdNamespace());
  const auto QType = hasCanonicalType(hasDeclaration(
      cxxRecordDecl(
          anyOf(EnableSharedFromThis.bind("enable_rec"),
                cxxRecordDecl(hasAnyBase(cxxBaseSpecifier(
                    isPublic(), hasType(hasCanonicalType(
                                    hasDeclaration(EnableSharedFromThis))))))))
          .bind("base_rec")));
  Finder->addMatcher(
      cxxRecordDecl(
          hasDirectBase(cxxBaseSpecifier(unless(isPublic()), hasType(QType))
                            .bind("base")))
          .bind("derived"),
      this);
}

void IncorrectEnableSharedFromThisCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BaseSpec = Result.Nodes.getNodeAs<CXXBaseSpecifier>("base");
  const auto *Base = Result.Nodes.getNodeAs<CXXRecordDecl>("base_rec");
  const auto *Derived = Result.Nodes.getNodeAs<CXXRecordDecl>("derived");
  const bool IsEnableSharedFromThisDirectBase =
      Result.Nodes.getNodeAs<CXXRecordDecl>("enable_rec") == Base;
  const SourceRange ReplacementRange = BaseSpec->getSourceRange();
  const std::string ReplacementString =
      // BaseSpec->getType().getAsString() results in
      // std::enable_shared_from_this<ClassName> or
      // alias/typedefs of std::enable_shared_from_this<ClassName>
      "public " + BaseSpec->getType().getAsString();
  const FixItHint Hint =
      FixItHint::CreateReplacement(ReplacementRange, ReplacementString);
  diag(Derived->getLocation(),
       "%2 is not publicly inheriting from "
       "%select{%1 which inherits from |}0'std::enable_shared_"
       "from_this', "
       "which will cause unintended behaviour "
       "when using 'shared_from_this'; make the inheritance "
       "public",
       DiagnosticIDs::Warning)
      << IsEnableSharedFromThisDirectBase << Base << Derived << Hint;
}

} // namespace clang::tidy::bugprone
