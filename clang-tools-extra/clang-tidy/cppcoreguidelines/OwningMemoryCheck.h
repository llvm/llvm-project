//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_OWNING_MEMORY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_OWNING_MEMORY_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::cppcoreguidelines {

/// Checks for common use cases for gsl::owner and enforces the unique owner
/// nature of it whenever possible.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/owning-memory.html
class OwningMemoryCheck : public ClangTidyCheck {
public:
  OwningMemoryCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        LegacyResourceProducers(Options.get(
            "LegacyResourceProducers", "::malloc;::aligned_alloc;::realloc;"
                                       "::calloc;::fopen;::freopen;::tmpfile")),
        LegacyResourceConsumers(Options.get(
            "LegacyResourceConsumers", "::free;::realloc;::freopen;::fclose")) {
  }
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }

  /// Make configuration of checker discoverable.
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  bool handleDeletion(const ast_matchers::BoundNodes &Nodes);
  bool handleLegacyConsumers(const ast_matchers::BoundNodes &Nodes);
  bool handleExpectedOwner(const ast_matchers::BoundNodes &Nodes);
  bool handleAssignmentAndInit(const ast_matchers::BoundNodes &Nodes);
  bool handleAssignmentFromNewOwner(const ast_matchers::BoundNodes &Nodes);
  bool handleReturnValues(const ast_matchers::BoundNodes &Nodes);
  bool handleOwnerMembers(const ast_matchers::BoundNodes &Nodes);

  /// List of old C-style functions that create resources.
  /// Defaults to
  /// `::malloc;::aligned_alloc;::realloc;::calloc;::fopen;::freopen;::tmpfile`.
  const StringRef LegacyResourceProducers;
  /// List of old C-style functions that consume or release resources.
  /// Defaults to `::free;::realloc;::freopen;::fclose`.
  const StringRef LegacyResourceConsumers;
};

} // namespace clang::tidy::cppcoreguidelines

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_OWNING_MEMORY_H
