//===--- UseToStringCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USE_TO_STRING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USE_TO_STRING_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace boost {

/// Finds calls to ``boost::lexical_cast<std::string>`` and
/// ``boost::lexical_cast<std::wstring>`` and replaces them with
/// ``std::to_string`` and ``std::to_wstring`` calls.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/boost-use-to-string.html
class UseToStringCheck : public ClangTidyCheck {
public:
  UseToStringCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace boost
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USE_TO_STRING_H
