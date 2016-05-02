//===--- ProTypeVarargCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_VARARG_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_VARARG_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// This check flags all calls to c-style variadic functions and all use
/// of va_arg.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-type-vararg.html
class ProTypeVarargCheck : public ClangTidyCheck {
public:
  ProTypeVarargCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_VARARG_H
