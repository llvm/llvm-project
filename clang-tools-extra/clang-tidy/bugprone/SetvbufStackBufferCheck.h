//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SETVBUFSTACKBUFFERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SETVBUFSTACKBUFFERCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Warns when setvbuf() is called with a stack-allocated buffer, which leads to
/// undefined behavior if the buffer's lifetime ends before the stream is closed
/// or reassigned.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/setvbuf-stack-buffer.html
class SetvbufStackBufferCheck : public ClangTidyCheck {
public:
  SetvbufStackBufferCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_SETVBUFSTACKBUFFERCHECK_H
