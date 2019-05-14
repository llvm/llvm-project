//===--- MacroParenthesesCheck.h - clang-tidy--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MACROPARENTHESESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MACROPARENTHESESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Finds macros that can have unexpected behaviour due to missing parentheses.
///
/// Macros are expanded by the preprocessor as-is. As a result, there can be
/// unexpected behaviour; operators may be evaluated in unexpected order and
/// unary operators may become binary operators, etc.
///
/// When the replacement list has an expression, it is recommended to surround
/// it with parentheses. This ensures that the macro result is evaluated
/// completely before it is used.
///
/// It is also recommended to surround macro arguments in the replacement list
/// with parentheses. This ensures that the argument value is calculated
/// properly.
class MacroParenthesesCheck : public ClangTidyCheck {
public:
  MacroParenthesesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_MACROPARENTHESESCHECK_H
