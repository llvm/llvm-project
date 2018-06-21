//===--- MultipleInheritanceCheck.h - clang-tidy-----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_MULTIPLE_INHERITANCE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_MULTIPLE_INHERITANCE_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace fuchsia {

/// Mulitple implementation inheritance is discouraged.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/fuchsia-multiple-inheritance.html
class MultipleInheritanceCheck : public ClangTidyCheck {
public:
  MultipleInheritanceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void onEndOfTranslationUnit() override { InterfaceMap.clear(); }

private:
  void addNodeToInterfaceMap(const CXXRecordDecl *Node, bool isInterface);
  bool getInterfaceStatus(const CXXRecordDecl *Node, bool &isInterface) const;
  bool isCurrentClassInterface(const CXXRecordDecl *Node) const;
  bool isInterface(const CXXRecordDecl *Node);

  // Contains the identity of each named CXXRecord as an interface.  This is
  // used to memoize lookup speeds and improve performance from O(N^2) to O(N),
  // where N is the number of classes.
  llvm::StringMap<bool> InterfaceMap;
};

} // namespace fuchsia
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FUCHSIA_MULTIPLE_INHERITANCE_H
