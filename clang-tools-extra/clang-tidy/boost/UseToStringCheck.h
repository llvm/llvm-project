//===--- UseToStringCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USE_TO_STRING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BOOST_USE_TO_STRING_H

#include "../ClangTidy.h"

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
