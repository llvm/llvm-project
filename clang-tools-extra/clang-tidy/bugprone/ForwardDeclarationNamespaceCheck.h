//===--- ForwardDeclarationNamespaceCheck.h - clang-tidy --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FORWARDDECLARATIONNAMESPACECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FORWARDDECLARATIONNAMESPACECHECK_H

#include "../ClangTidyCheck.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>
#include <vector>

namespace clang {
namespace tidy {
namespace bugprone {

/// Checks if an unused forward declaration is in a wrong namespace.
///
/// The check inspects all unused forward declarations and checks if there is
/// any declaration/definition with the same name, which could indicate
/// that the forward declaration is potentially in a wrong namespace.
///
/// \code
///   namespace na { struct A; }
///   namespace nb { struct A {} };
///   nb::A a;
///   // warning : no definition found for 'A', but a definition with the same
///   name 'A' found in another namespace 'nb::'
/// \endcode
///
/// This check can only generate warnings, but it can't suggest fixes at this
/// point.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/forward-declaration-namespace.html
class ForwardDeclarationNamespaceCheck : public ClangTidyCheck {
public:
  ForwardDeclarationNamespaceCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  llvm::StringMap<std::vector<const CXXRecordDecl *>> DeclNameToDefinitions;
  llvm::StringMap<std::vector<const CXXRecordDecl *>> DeclNameToDeclarations;
  llvm::SmallPtrSet<const Type *, 16> FriendTypes;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_FORWARDDECLARATIONNAMESPACECHECK_H
