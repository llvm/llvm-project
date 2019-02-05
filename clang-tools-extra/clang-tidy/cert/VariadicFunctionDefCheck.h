//===--- VariadicFunctionDefCheck.h - clang-tidy-----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_VARIADICFUNCTIONDEF_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_VARIADICFUNCTIONDEF_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cert {

/// Guards against any C-style variadic function definitions (not declarations).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-dcl50-cpp.html
class VariadicFunctionDefCheck : public ClangTidyCheck {
public:
  VariadicFunctionDefCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_VARIADICFUNCTIONDEF_H
