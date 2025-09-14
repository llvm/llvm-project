//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DUPLICATE_INCLUDE_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DUPLICATE_INCLUDE_CHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// \brief Find and remove duplicate #include directives.
///
/// Only consecutive include directives without any other preprocessor
/// directives between them are analyzed.
class DuplicateIncludeCheck : public ClangTidyCheck {
public:
  DuplicateIncludeCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_DUPLICATE_INCLUDE_CHECK_H
