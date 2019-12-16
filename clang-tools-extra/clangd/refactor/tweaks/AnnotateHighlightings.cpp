//===--- AnnotateHighlightings.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SemanticHighlighting.h"
#include "refactor/Tweak.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {
namespace {

/// Annotate all highlighting tokens in the current file. This is a hidden tweak
/// which is used to debug semantic highlightings.
/// Before:
///   void f() { int abc; }
///   ^^^^^^^^^^^^^^^^^^^^^
/// After:
///   void /* entity.name.function.cpp */ f() { int /* variable.cpp */ abc; }
class AnnotateHighlightings : public Tweak {
public:
  const char *id() const override final;

  bool prepare(const Selection &Inputs) override { return true; }
  Expected<Effect> apply(const Selection &Inputs) override;

  std::string title() const override { return "Annotate highlighting tokens"; }
  Intent intent() const override { return Refactor; }
  bool hidden() const override { return true; }
};
REGISTER_TWEAK(AnnotateHighlightings)

Expected<Tweak::Effect> AnnotateHighlightings::apply(const Selection &Inputs) {
  const Decl *CommonDecl = nullptr;
  for (auto N = Inputs.ASTSelection.commonAncestor(); N && !CommonDecl;
       N = N->Parent)
    CommonDecl = N->ASTNode.get<Decl>();

  std::vector<HighlightingToken> HighlightingTokens;
  if (!CommonDecl) {
    // Now we hit the TUDecl case where commonAncestor() returns null
    // intendedly. We only annotate tokens in the main file, so use the default
    // traversal scope (which is the top level decls of the main file).
    HighlightingTokens = getSemanticHighlightings(*Inputs.AST);
  } else {
    // Store the existing scopes.
    const auto &BackupScopes = Inputs.AST->getASTContext().getTraversalScope();
    // Narrow the traversal scope to the selected node.
    Inputs.AST->getASTContext().setTraversalScope(
        {const_cast<Decl *>(CommonDecl)});
    HighlightingTokens = getSemanticHighlightings(*Inputs.AST);
    // Restore the traversal scope.
    Inputs.AST->getASTContext().setTraversalScope(BackupScopes);
  }
  auto &SM = Inputs.AST->getSourceManager();
  tooling::Replacements Result;
  llvm::StringRef FilePath = SM.getFilename(Inputs.Cursor);
  for (const auto &Token : HighlightingTokens) {
    assert(Token.R.start.line == Token.R.end.line &&
           "Token must be at the same line");
    auto InsertOffset = positionToOffset(Inputs.Code, Token.R.start);
    if (!InsertOffset)
      return InsertOffset.takeError();

    auto InsertReplacement = tooling::Replacement(
        FilePath, *InsertOffset, 0,
        ("/* " + toTextMateScope(Token.Kind) + " */").str());
    if (auto Err = Result.add(InsertReplacement))
      return std::move(Err);
  }
  return Effect::mainFileEdit(SM, std::move(Result));
}

} // namespace
} // namespace clangd
} // namespace clang
