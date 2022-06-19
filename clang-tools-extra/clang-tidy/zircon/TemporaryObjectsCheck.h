//===--- TemporaryObjectsCheck.h - clang-tidy------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ZIRCON_TEMPORARYOBJECTSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ZIRCON_TEMPORARYOBJECTSCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/OptionsUtils.h"

namespace clang {
namespace tidy {
namespace zircon {

/// Construction of specific temporary objects in the Zircon kernel is
/// discouraged.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/zircon/temporary-objects.html
class TemporaryObjectsCheck : public ClangTidyCheck {
public:
  TemporaryObjectsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        Names(utils::options::parseStringList(Options.get("Names", ""))) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::vector<StringRef> Names;
};

} // namespace zircon
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ZIRCON_TEMPORARYOBJECTSCHECK_H
