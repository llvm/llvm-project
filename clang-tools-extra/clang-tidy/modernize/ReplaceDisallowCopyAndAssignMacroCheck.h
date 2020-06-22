//===--- ReplaceDisallowCopyAndAssignMacroCheck.h - clang-tidy --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACEDISALLOWCOPYANDASSIGNMACROCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACEDISALLOWCOPYANDASSIGNMACROCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// This check finds macro expansions of ``DISALLOW_COPY_AND_ASSIGN(Type)`` and
/// replaces them with a deleted copy constructor and a deleted assignment
/// operator.
///
/// Before:
/// ~~~{.cpp}
///   class Foo {
///   private:
///     DISALLOW_COPY_AND_ASSIGN(Foo);
///   };
/// ~~~
///
/// After:
/// ~~~{.cpp}
///   class Foo {
///   private:
///     Foo(const Foo &) = delete;
///     const Foo &operator=(const Foo &) = delete;
///   };
/// ~~~
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-replace-disallow-copy-and-assign-macro.html
class ReplaceDisallowCopyAndAssignMacroCheck : public ClangTidyCheck {
public:
  ReplaceDisallowCopyAndAssignMacroCheck(StringRef Name,
                                         ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  const std::string &getMacroName() const { return MacroName; }

private:
  const std::string MacroName;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_REPLACEDISALLOWCOPYANDASSIGNMACROCHECK_H
