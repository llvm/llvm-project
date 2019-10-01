//===--- SIMDIntrinsicsCheck.h - clang-tidy----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SIMD_INTRINSICS_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SIMD_INTRINSICS_CHECK_H

#include "../ClangTidyCheck.h"

#include "llvm/ADT/SmallString.h"

namespace clang {
namespace tidy {
namespace portability {

/// Find SIMD intrinsics calls and suggest std::experimental::simd alternatives.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability-simd-intrinsics.html
class SIMDIntrinsicsCheck : public ClangTidyCheck {
public:
  SIMDIntrinsicsCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

 private:
  llvm::SmallString<32> Std;
  const bool Suggest;
};

} // namespace portability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SIMD_INTRINSICS_CHECK_H
