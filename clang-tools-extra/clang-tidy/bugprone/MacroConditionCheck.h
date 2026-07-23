//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MACROCONDITIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MACROCONDITIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Warns about inconsistent macro usage in preprocessor conditions.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/bugprone-macro-condition.html
class MacroConditionCheck : public ClangTidyCheck {
public:
  using ClangTidyCheck::ClangTidyCheck;

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MACROCONDITIONCHECK_H
