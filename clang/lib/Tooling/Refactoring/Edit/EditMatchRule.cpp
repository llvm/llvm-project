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
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/SourceCode.h"

using namespace clang;
using namespace tooling;
using namespace transformer;

Expected<EditMatchRule>
EditMatchRule::initiate(RefactoringRuleContext &Context,
                        ast_matchers::MatchFinder::MatchResult R,
                        transformer::EditGenerator EG) {
  return EditMatchRule(std::move(R), std::move(EG));
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
  Expected<SmallVector<transformer::Edit, 1>> Edits = EditGenerator(Result);

  if (!Edits) {
    return std::move(Edits.takeError());
  }
  if (Edits->empty())
    return Context.createDiagnosticError(
        diag::err_refactor_invalid_edit_generator);

  AtomicChange Change(SM, Edits->front().Range.getBegin());
  {
    for (const auto &Edit : *Edits) {
      switch (Edit.Kind) {
      case EditKind::Range:
        if (auto Err = Change.replace(SM, std::move(Edit.Range),
                                      std::move(Edit.Replacement))) {
          return std::move(Err);
        }
        break;
      case EditKind::AddInclude:
        Change.addHeader(std::move(Edit.Replacement));
        break;
      }
    }
  }

  return AtomicChanges{std::move(Change)};
}
