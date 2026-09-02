//===--- CallGuardInitCheck.h - clang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PYBIND_CALLGUARDINITCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PYBIND_CALLGUARDINITCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::pybind {

/// Finds pybind11 `py::init` definitions guarded by
/// `py::call_guard<py::gil_scoped_release>`.
///
/// Using `py::call_guard<py::gil_scoped_release>()` on `py::init(...)` keeps
/// the Python GIL released during constructor trampoline execution, causing
/// instance construction and Python object initialization to run without the
/// GIL.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/pybind/call-guard-init.html
class CallGuardInitCheck : public ClangTidyCheck {
public:
  CallGuardInitCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
};

} // namespace clang::tidy::pybind

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PYBIND_CALLGUARDINITCHECK_H
