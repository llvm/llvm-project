//===--- RestrictSystemLibcHeadersCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVMLIBC_RESTRICTSYSTEMLIBCHEADERSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVMLIBC_RESTRICTSYSTEMLIBCHEADERSCHECK_H

#include "../ClangTidyCheck.h"
#include "../portability/RestrictSystemIncludesCheck.h"

namespace clang {
namespace tidy {
namespace llvm_libc {

/// Warns of accidental inclusions of system libc headers that aren't
/// compiler provided.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/llvmlibc/restrict-system-libc-headers.html
class RestrictSystemLibcHeadersCheck
    : public portability::RestrictSystemIncludesCheck {
public:
  RestrictSystemLibcHeadersCheck(StringRef Name, ClangTidyContext *Context)
      : portability::RestrictSystemIncludesCheck(Name, Context, "-*") {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace llvm_libc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVMLIBC_RESTRICTSYSTEMLIBCHEADERSCHECK_H
