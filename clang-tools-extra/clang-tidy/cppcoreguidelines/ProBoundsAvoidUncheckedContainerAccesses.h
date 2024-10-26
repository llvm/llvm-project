//===--- ProBoundsAvoidUncheckedContainerAccesses.h - clang-tidy *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_BOUNDS_AVOID_UNCHECKED_CONTAINER_ACCESSES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_BOUNDS_AVOID_UNCHECKED_CONTAINER_ACCESSES_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Flags calls to operator[] in STL containers and suggests replacing it with
/// safe alternatives.
///
/// See
/// https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#slcon3-avoid-bounds-errors
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/pro-bounds-avoid-unchecked-container-accesses.html
class ProBoundsAvoidUncheckedContainerAccesses : public ClangTidyCheck {
public:
  ProBoundsAvoidUncheckedContainerAccesses(StringRef Name,
                                           ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  enum SubscriptFixModes { None, At, Function };

private:
  // A list of class names that are excluded from the warning
  std::vector<llvm::StringRef> SubscriptExcludedClasses;
  // Setting which fix to suggest
  SubscriptFixModes SubscriptFixMode;
  llvm::StringRef SubscriptFixFunction;
};
} // namespace clang::tidy::cppcoreguidelines

namespace clang::tidy {
template <>
struct OptionEnumMapping<
    cppcoreguidelines::ProBoundsAvoidUncheckedContainerAccesses::
        SubscriptFixModes> {
  static ArrayRef<
      std::pair<cppcoreguidelines::ProBoundsAvoidUncheckedContainerAccesses::
                    SubscriptFixModes,
                StringRef>>
  getEnumMapping();
};
} // namespace clang::tidy
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_BOUNDS_AVOID_UNCHECKED_CONTAINER_ACCESSES_H
