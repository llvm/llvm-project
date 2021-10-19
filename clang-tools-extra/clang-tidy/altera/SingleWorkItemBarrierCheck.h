//===--- SingleWorkItemBarrierCheck.h - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_SINGLE_WORK_ITEM_BARRIER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_SINGLE_WORK_ITEM_BARRIER_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace altera {

/// Detects OpenCL kernel functions that call a barrier but do not call an
/// ID-function function. These functions will be treated as single work-item
/// kernels, which may be inefficient or cause an error.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/opencl-single-work-item-barrier.html
class SingleWorkItemBarrierCheck : public ClangTidyCheck {
  const unsigned AOCVersion;

public:
  SingleWorkItemBarrierCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        AOCVersion(Options.get("AOCVersion", 1600U)) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
};

} // namespace altera
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_SINGLE_WORK_ITEM_BARRIER_H
