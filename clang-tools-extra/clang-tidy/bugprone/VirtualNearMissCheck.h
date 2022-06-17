//===--- VirtualNearMissCheck.h - clang-tidy---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_VIRTUAL_NEAR_MISS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_VIRTUAL_NEAR_MISS_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks for near miss of virtual methods.
///
/// For a method in a derived class, this check looks for virtual method with a
/// very similar name and an identical signature defined in a base class.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/virtual-near-miss.html
class VirtualNearMissCheck : public ClangTidyCheck {
public:
  VirtualNearMissCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  /// Check if the given method is possible to be overridden by some other
  /// method. Operators and destructors are excluded.
  ///
  /// Results are memoized in PossibleMap.
  bool isPossibleToBeOverridden(const CXXMethodDecl *BaseMD);

  /// Check if the given base method is overridden by some methods in the given
  /// derived class.
  ///
  /// Results are memoized in OverriddenMap.
  bool isOverriddenByDerivedClass(const CXXMethodDecl *BaseMD,
                                  const CXXRecordDecl *DerivedRD);

  /// Key: the unique ID of a method.
  /// Value: whether the method is possible to be overridden.
  llvm::DenseMap<const CXXMethodDecl *, bool> PossibleMap;

  /// Key: <unique ID of base method, name of derived class>
  /// Value: whether the base method is overridden by some method in the derived
  /// class.
  llvm::DenseMap<std::pair<const CXXMethodDecl *, const CXXRecordDecl *>, bool>
      OverriddenMap;

  const unsigned EditDistanceThreshold = 1;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_VIRTUAL_NEAR_MISS_H
