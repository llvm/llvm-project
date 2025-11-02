//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_USENEWMLIROPBUILDERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_USENEWMLIROPBUILDERCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Checks for uses of MLIR's old/to be deprecated `OpBuilder::create<T>` form
/// and suggests using `T::create` instead.
class UseNewMlirOpBuilderCheck : public utils::TransformerClangTidyCheck {
public:
  UseNewMlirOpBuilderCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return getLangOpts().CPlusPlus;
  }
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_USENEWMLIROPBUILDERCHECK_H
