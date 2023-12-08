//===--- TypePromotionInMathFnCheck.h - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_TYPE_PROMOTION_IN_MATH_FN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_TYPE_PROMOTION_IN_MATH_FN_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::performance {

/// Finds calls to C math library functions with implicit float to double
/// promotions.
///
/// For example, warns on ::sin(0.f), because this function's parameter is a
/// double.  You probably meant to call std::sin(0.f) (in C++), or sinf(0.f) (in
/// C).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/type-promotion-in-math-fn.html
class TypePromotionInMathFnCheck : public ClangTidyCheck {
public:
  TypePromotionInMathFnCheck(StringRef Name, ClangTidyContext *Context);

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  utils::IncludeInserter IncludeInserter;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_TYPE_PROMOTION_IN_MATH_FN_H
