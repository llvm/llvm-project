//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_CLEANUPCTADCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_CLEANUPCTADCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::abseil {

/// Suggests switching the initialization pattern of `absl::Cleanup`
/// instances from the factory function to class template argument
/// deduction (CTAD), in C++17 and higher.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/abseil/cleanup-ctad.html
class CleanupCtadCheck : public utils::TransformerClangTidyCheck {
public:
  CleanupCtadCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }
};

} // namespace clang::tidy::abseil

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ABSEIL_CLEANUPCTADCHECK_H
