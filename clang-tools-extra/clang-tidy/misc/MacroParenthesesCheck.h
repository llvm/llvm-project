//===--- MacroParenthesesCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MACRO_PARENTHESES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MACRO_PARENTHESES_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

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
  void registerPPCallbacks(CompilerInstance &Compiler) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MACRO_PARENTHESES_H
