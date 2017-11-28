//===--- MoveConstantArgumentCheck.h - clang-tidy -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace misc {

/// Find casts of calculation results to bigger type. Typically from int to
///
/// There is one option:
///
///   - `CheckTriviallyCopyableMove`: Whether to check for trivially-copyable
//      types as their objects are not moved but copied. Enabled by default.
class MoveConstantArgumentCheck : public ClangTidyCheck {
public:
  MoveConstantArgumentCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        CheckTriviallyCopyableMove(
            Options.get("CheckTriviallyCopyableMove", true)) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const bool CheckTriviallyCopyableMove;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_MOVECONSTANTARGUMENTCHECK_H
