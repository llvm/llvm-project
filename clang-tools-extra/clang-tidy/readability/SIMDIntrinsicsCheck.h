//===--- SIMDIntrinsicsCheck.h - clang-tidy----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SIMD_INTRINSICS_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SIMD_INTRINSICS_CHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Find SIMD intrinsics calls and suggest std::experimental::simd alternatives.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-simd-intrinsics.html
class SIMDIntrinsicsCheck : public ClangTidyCheck {
public:
  SIMDIntrinsicsCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

 private:
  const bool Suggest;
  StringRef Std;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SIMD_INTRINSICS_CHECK_H
