//===--- ConvertMemberFunctionsToStatic.h - clang-tidy ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONVERTMEMFUNCTOSTATIC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONVERTMEMFUNCTOSTATIC_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// This check finds C++ class methods than can be made static
/// because they don't use the 'this' pointer.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/
/// readability-convert-member-functions-to-static.html
class ConvertMemberFunctionsToStatic : public ClangTidyCheck {
public:
  ConvertMemberFunctionsToStatic(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_CONVERTMEMFUNCTOSTATIC_H
