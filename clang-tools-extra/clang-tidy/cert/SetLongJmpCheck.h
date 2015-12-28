//===--- SetLongJmpCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SETLONGJMPCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SETLONGJMPCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {

/// Guards against use of setjmp/longjmp in C++ code
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-err52-cpp.html
class SetLongJmpCheck : public ClangTidyCheck {
public:
  SetLongJmpCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(CompilerInstance &Compiler) override;

  static const char DiagWording[];
};

} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_SETLONGJMPCHECK_H

