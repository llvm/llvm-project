//===--- IncludeOrderCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_INCLUDE_ORDER_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_INCLUDE_ORDER_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace llvm {

/// Checks the correct order of `#includes`.
///
/// See http://llvm.org/docs/CodingStandards.html#include-style
class IncludeOrderCheck : public ClangTidyCheck {
public:
  IncludeOrderCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace llvm
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_INCLUDE_ORDER_CHECK_H
