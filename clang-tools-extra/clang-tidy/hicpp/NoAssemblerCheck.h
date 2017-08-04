//===--- NoAssemblerCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_NO_ASSEMBLER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_NO_ASSEMBLER_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace hicpp {

/// Find assembler statements. No fix is offered.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/hicpp-no-assembler.html
class NoAssemblerCheck : public ClangTidyCheck {
public:
  NoAssemblerCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace hicpp
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HICPP_NO_ASSEMBLER_H
