//===--- UseConcisePreprocessorDirectivesCheck.h - clang-tidy ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USECONCISEPREPROCESSORDIRECTIVESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USECONCISEPREPROCESSORDIRECTIVESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Shortens `#if` preprocessor conditions:
///
/// #if  defined(MEOW) -> #ifdef  MEOW
/// #if !defined(MEOW) -> #ifndef MEOW
///
/// And, since C23 and C++23, shortens `#elif` conditions too:
///
/// #elif  defined(MEOW) -> #elifdef  MEOW
/// #elif !defined(MEOW) -> #elifndef MEOW
///
/// User-facing documentation:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-concise-preprocessor-directives.html
class UseConcisePreprocessorDirectivesCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USECONCISEPREPROCESSORDIRECTIVESCHECK_H
