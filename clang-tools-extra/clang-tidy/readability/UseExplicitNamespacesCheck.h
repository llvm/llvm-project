//===--- UseExplicitNamespacesCheck.h - clang-tidy --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEEXPLICITNAMESPACESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEEXPLICITNAMESPACESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// FIXME: Write a short description.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/readability/use-explicit-namespaces.html
class UseExplicitNamespacesCheck : public ClangTidyCheck {
public:
  UseExplicitNamespacesCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }

  IdentifierInfo *
  getTypeNestedNameIdentfierRecursive(const Type *type,
                                      std::string &nestedTypeInfo);
  bool matchesNamespaceLimits(
      const std::vector<const DeclContext *> &targetContextVector);

  void diagOut(const SourceLocation &sourcePosition,
               const std::string &message);

  void processTransform(NestedNameSpecifierLoc nestedName,
                        const SourceLocation &sourcePosition,
                        const NamedDecl *target,
                        const DeclContext *referenceContext, bool usingShadow,
                        const std::string &context);

  void processTypePiecesRecursive(NestedNameSpecifierLoc nestedName,
                                  const TypeLoc &typeLoc,
                                  const DeclContext *declContext,
                                  const std::string &context);

  void processTypePieces(TypeSourceInfo *typeSourceInfo,
                         const DeclContext *declContext,
                         const std::string &context);

private:
  const StringRef limitToPattern;
  std::vector<std::string> limitToPatternVector;
  bool onlyExpandUsingNamespace;
  const unsigned diagnosticLevel;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_USEEXPLICITNAMESPACESCHECK_H
