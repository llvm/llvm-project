//===--- CppInitClassMembersCheck.h - clang-tidy ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_CPPINITCLASSMEMBERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_CPPINITCLASSMEMBERSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::google {

/// Checks that class members are initialized in constructors (implicitly or
/// explicitly). Reports constructors or classes where class members are not
/// initialized. The goal of this checker is to eliminate UUM (Use of
/// Uninitialized Memory) bugs caused by uninitialized class members.
///
/// This checker is different from ProTypeMemberInitCheck in that this checker
/// attempts to eliminate UUMs as a bug class, at the expense of false
/// positives.
///
/// This checker is WIP. We are incrementally adding features and increasing
/// coverage until we get to a shape that is acceptable.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/google/cpp-init-class-members.html
class CppInitClassMembersCheck : public ClangTidyCheck {
public:
  CppInitClassMembersCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus && !LangOpts.ObjC;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  // Issue a diagnostic for any constructor that does not initialize all member
  // variables. If the record does not have a constructor (`Ctor` is `nullptr`),
  // the diagnostic is for the record.
  void checkMissingMemberInitializer(ASTContext &Context,
                                     const CXXRecordDecl &ClassDecl,
                                     const CXXConstructorDecl *Ctor);
};

} // namespace clang::tidy::google

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_GOOGLE_CPPINITCLASSMEMBERSCHECK_H