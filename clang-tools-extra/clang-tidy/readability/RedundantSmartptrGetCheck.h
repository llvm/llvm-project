//===--- RedundantSmartptrGetCheck.h - clang-tidy ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTSMARTPTRGETCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTSMARTPTRGETCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace readability {

/// Find and remove redundant calls to smart pointer's `.get()` method.
///
/// Examples:
///
/// \code
///   ptr.get()->Foo()  ==>  ptr->Foo()
///   *ptr.get()  ==>  *ptr
///   *ptr->get()  ==>  **ptr
/// \endcode
class RedundantSmartptrGetCheck : public ClangTidyCheck {
public:
  RedundantSmartptrGetCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", 1) != 0) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool IgnoreMacros;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_REDUNDANTSMARTPTRGETCHECK_H
