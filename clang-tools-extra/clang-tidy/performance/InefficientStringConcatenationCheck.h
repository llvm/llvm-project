//===--- InefficientStringConcatenationCheck.h - clang-tidy-----------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTSTRINGCONCATENATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTSTRINGCONCATENATION_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace performance {

/// This check is to warn about the performance overhead arising from
/// concatenating strings, using the operator+, instead of operator+=.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance-inefficient-string-concatenation.html
class InefficientStringConcatenationCheck : public ClangTidyCheck {
public:
  InefficientStringConcatenationCheck(StringRef Name,
                                      ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  const bool StrictMode;
};

} // namespace performance
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_INEFFICIENTSTRINGCONCATENATION_H
