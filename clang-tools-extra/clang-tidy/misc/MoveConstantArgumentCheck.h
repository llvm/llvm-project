//===--- MoveConstandArgumentCheck.h - clang-tidy -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONTANTARGUMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONTANTARGUMENTCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

class MoveConstantArgumentCheck : public ClangTidyCheck {
public:
  MoveConstantArgumentCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONTANTARGUMENTCHECK_H
