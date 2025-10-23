//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOIDUNCONDITIONALPREPROCESSORIFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOIDUNCONDITIONALPREPROCESSORIFCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Finds code blocks that are constantly enabled or disabled in preprocessor
/// directives by analyzing `#if` conditions, such as `#if 0` and `#if 1`, etc.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/avoid-unconditional-preprocessor-if.html
class AvoidUnconditionalPreprocessorIfCheck : public ClangTidyCheck {
public:
  AvoidUnconditionalPreprocessorIfCheck(StringRef Name,
                                        ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_AVOIDUNCONDITIONALPREPROCESSORIFCHECK_H
