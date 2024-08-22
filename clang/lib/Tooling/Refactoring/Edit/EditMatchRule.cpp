//===--- EditMatchRule.cpp - Clang refactoring library --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements the "edit-match" refactoring rule that can edit matcher results
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Edit/EditMatchRule.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/DiagnosticRefactoring.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/SourceCode.h"

using namespace clang;
using namespace tooling;
using namespace transformer;

Expected<EditMatchRule>
EditMatchRule::initiate(RefactoringRuleContext &Context,
                        ast_matchers::MatchFinder::MatchResult R, ASTEdit AE) {
  return EditMatchRule(std::move(R), std::move(AE));
}

const RefactoringDescriptor &EditMatchRule::describe() {
  static const RefactoringDescriptor Descriptor = {
      "edit-match",
      "Edit Match",
      "Edits match result source code",
  };
  return Descriptor;
}

Expected<AtomicChanges>
EditMatchRule::createSourceReplacements(RefactoringRuleContext &Context) {
  ASTContext &AST = Context.getASTContext();
  SourceManager &SM = AST.getSourceManager();

  Expected<CharSourceRange> Range = Edit.TargetRange(Result);
  if (!Range)
    return std::move(Range.takeError());
  std::optional<CharSourceRange> EditRange =
      getFileRangeForEdit(std::move(*Range), AST);
  if (!EditRange)
    return Context.createDiagnosticError(diag::err_refactor_no_ast_edit);

  AtomicChange Change(SM, EditRange->getBegin());
  {
    auto Replacement = Edit.Replacement->eval(Result);
    if (!Replacement)
      return std::move(Replacement.takeError());

    switch (Edit.Kind) {
    case EditKind::Range:
      if (auto Err = Change.replace(SM, std::move(*EditRange),
                                    std::move(*Replacement))) {
        return std::move(Err);
      }
      break;
    case EditKind::AddInclude:
      Change.addHeader(std::move(*Replacement));
      break;
    }
  }

  return AtomicChanges{std::move(Change)};
}
