//===- InconsistentDeclarationParameterNameCheck.h - clang-tidy-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_INCONSISTENT_DECLARATION_PARAMETER_NAME_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_INCONSISTENT_DECLARATION_PARAMETER_NAME_H

#include "../ClangTidy.h"

#include "llvm/ADT/DenseSet.h"

namespace clang {
namespace tidy {
namespace readability {

/// \brief Checks for declarations of functions which differ in parameter names.
///
/// For detailed documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability-inconsistent-declaration-parameter-name.html
///
class InconsistentDeclarationParameterNameCheck : public ClangTidyCheck {
public:
  InconsistentDeclarationParameterNameCheck(StringRef Name,
                                            ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context),
        IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", 1) != 0),
        Strict(Options.getLocalOrGlobal("Strict", 0) != 0) {}

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void markRedeclarationsAsVisited(const FunctionDecl *FunctionDeclaration);

  llvm::DenseSet<const FunctionDecl *> VisitedDeclarations;
  const bool IgnoreMacros;
  const bool Strict;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_INCONSISTENT_DECLARATION_PARAMETER_NAME_H
