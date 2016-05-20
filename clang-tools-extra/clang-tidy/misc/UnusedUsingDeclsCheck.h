//===--- UnusedUsingDeclsCheck.h - clang-tidy--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_USING_DECLS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_USING_DECLS_H

#include "../ClangTidy.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>

namespace clang {
namespace tidy {
namespace misc {

/// Finds unused using declarations.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-unused-using-decls.html
class UnusedUsingDeclsCheck : public ClangTidyCheck {
public:
  UnusedUsingDeclsCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  void removeFromFoundDecls(const Decl *D);

  struct UsingDeclContext {
    explicit UsingDeclContext(const UsingDecl *FoundUsingDecl)
        : FoundUsingDecl(FoundUsingDecl), IsUsed(false) {}
    llvm::SmallPtrSet<const Decl *, 4> UsingTargetDecls;
    const UsingDecl *FoundUsingDecl;
    CharSourceRange UsingDeclRange;
    bool IsUsed;
  };

  std::vector<UsingDeclContext> Contexts;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_UNUSED_USING_DECLS_H
