//===--- TemporaryObjectsCheck.h - clang-tidy------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ZIRCON_TEMPORARYOBJECTSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ZIRCON_TEMPORARYOBJECTSCHECK_H

#include "../ClangTidy.h"
#include "../utils/OptionsUtils.h"

namespace clang {
namespace tidy {
namespace zircon {

/// Construction of specific temporary objects in the Zircon kernel is
/// discouraged.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/zircon-temporary-objects.html
class TemporaryObjectsCheck : public ClangTidyCheck {
public:
  TemporaryObjectsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        Names(utils::options::parseStringList(Options.get("Names", ""))) {}
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  std::vector<std::string> Names;
};

} // namespace zircon
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ZIRCON_TEMPORARYOBJECTSCHECK_H
