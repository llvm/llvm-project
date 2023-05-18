//===--- LoopConvertCheck.h - clang-tidy-------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "LoopConvertUtils.h"

namespace clang::tidy::modernize {

class LoopConvertCheck : public ClangTidyCheck {
public:
  LoopConvertCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  struct RangeDescriptor {
    RangeDescriptor() = default;
    bool ContainerNeedsDereference = false;
    bool DerefByConstRef = false;
    bool DerefByValue = false;
    std::string ContainerString;
    QualType ElemType;
    bool NeedsReverseCall = false;
  };

  void getAliasRange(SourceManager &SM, SourceRange &DeclRange);

  void doConversion(ASTContext *Context, const VarDecl *IndexVar,
                    const ValueDecl *MaybeContainer, const UsageResult &Usages,
                    const DeclStmt *AliasDecl, bool AliasUseRequired,
                    bool AliasFromForInit, const ForStmt *Loop,
                    RangeDescriptor Descriptor);

  StringRef getContainerString(ASTContext *Context, const ForStmt *Loop,
                               const Expr *ContainerExpr);

  void getArrayLoopQualifiers(ASTContext *Context,
                              const ast_matchers::BoundNodes &Nodes,
                              const Expr *ContainerExpr,
                              const UsageResult &Usages,
                              RangeDescriptor &Descriptor);

  void getIteratorLoopQualifiers(ASTContext *Context,
                                 const ast_matchers::BoundNodes &Nodes,
                                 RangeDescriptor &Descriptor);

  void determineRangeDescriptor(ASTContext *Context,
                                const ast_matchers::BoundNodes &Nodes,
                                const ForStmt *Loop, LoopFixerKind FixerKind,
                                const Expr *ContainerExpr,
                                const UsageResult &Usages,
                                RangeDescriptor &Descriptor);

  bool isConvertible(ASTContext *Context, const ast_matchers::BoundNodes &Nodes,
                     const ForStmt *Loop, LoopFixerKind FixerKind);

  StringRef getReverseFunction() const;
  StringRef getReverseHeader() const;

  std::unique_ptr<TUTrackingInfo> TUInfo;
  const unsigned long long MaxCopySize;
  const Confidence::Level MinConfidence;
  const VariableNamer::NamingStyle NamingStyle;
  utils::IncludeInserter Inserter;
  bool UseReverseRanges;
  const bool UseCxx20IfAvailable;
  std::string ReverseFunction;
  std::string ReverseHeader;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_LOOP_CONVERT_H
