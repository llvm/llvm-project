//===--- ConstReturnTypeCheck.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// For any function whose return type is const-qualified, suggests removal of
/// the `const` qualifier from that return type.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/const-return-type.html
class ConstReturnTypeCheck : public ClangTidyCheck {
 public:
   ConstReturnTypeCheck(StringRef Name, ClangTidyContext *Context)
       : ClangTidyCheck(Name, Context),
         IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}
   void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
   void registerMatchers(ast_matchers::MatchFinder *Finder) override;
   void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

 private:
   const bool IgnoreMacros;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONSTRETURNTYPECHECK_H
