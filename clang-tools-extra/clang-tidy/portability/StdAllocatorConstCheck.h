//===--- StdAllocatorConstT.h - clang-tidy -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_STDALLOCATORCONSTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_STDALLOCATORCONSTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace portability {

/// Report use of ``std::vector<const T>`` (and similar containers of const
/// elements). These are not allowed in standard C++ due to undefined
/// ``std::allocator<const T>``. They do not compile with libstdc++ or MSVC.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/portability-std-allocator-const.html
class StdAllocatorConstCheck : public ClangTidyCheck {
public:
  StdAllocatorConstCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace portability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PORTABILITY_STDALLOCATORCONSTCHECK_H
