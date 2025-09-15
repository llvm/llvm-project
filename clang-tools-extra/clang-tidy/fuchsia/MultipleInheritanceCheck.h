//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_MULTIPLE_INHERITANCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_MULTIPLE_INHERITANCE_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::fuchsia {

/// Multiple implementation inheritance is discouraged.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia/multiple-inheritance.html
class MultipleInheritanceCheck : public ClangTidyCheck {
public:
  MultipleInheritanceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void onEndOfTranslationUnit() override { InterfaceMap.clear(); }

private:
  void addNodeToInterfaceMap(const CXXRecordDecl *Node, bool IsInterface);
  bool getInterfaceStatus(const CXXRecordDecl *Node, bool &IsInterface) const;
  bool isCurrentClassInterface(const CXXRecordDecl *Node) const;
  bool isInterface(const CXXRecordDecl *Node);

  // Contains the identity of each named CXXRecord as an interface.  This is
  // used to memoize lookup speeds and improve performance from O(N^2) to O(N),
  // where N is the number of classes.
  llvm::StringMap<bool> InterfaceMap;
};

} // namespace clang::tidy::fuchsia

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_MULTIPLE_INHERITANCE_H
