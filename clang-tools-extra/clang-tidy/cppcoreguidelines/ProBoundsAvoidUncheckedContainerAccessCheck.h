//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PROBOUNDSAVOIDUNCHECKEDCONTAINERACCESSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PROBOUNDSAVOIDUNCHECKEDCONTAINERACCESSCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Flags calls to operator[] in STL containers and suggests replacing it with
/// safe alternatives.
///
/// See
/// https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#slcon3-avoid-bounds-errors
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/pro-bounds-avoid-unchecked-container-access.html
class ProBoundsAvoidUncheckedContainerAccessCheck : public ClangTidyCheck {
public:
  ProBoundsAvoidUncheckedContainerAccessCheck(StringRef Name,
                                              ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  enum FixModes { None, At, Function };

private:
  // A list of class names that are excluded from the warning
  std::vector<llvm::StringRef> ExcludedClasses;
  // Setting which fix to suggest
  FixModes FixMode;
  llvm::StringRef FixFunction;
  llvm::StringRef FixFunctionEmptyArgs;
};
} // namespace clang::tidy::cppcoreguidelines

namespace clang::tidy {
template <>
struct OptionEnumMapping<
    cppcoreguidelines::ProBoundsAvoidUncheckedContainerAccessCheck::FixModes> {
  static ArrayRef<std::pair<
      cppcoreguidelines::ProBoundsAvoidUncheckedContainerAccessCheck::FixModes,
      StringRef>>
  getEnumMapping();
};
} // namespace clang::tidy
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PROBOUNDSAVOIDUNCHECKEDCONTAINERACCESSCHECK_H
