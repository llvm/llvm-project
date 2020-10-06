//===--- ObjcLocalizeStringLiteral.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {

/// Wraps an Objective-C string literal with the NSLocalizedString macro.
/// Before:
///   @"description"
///   ^^^
/// After:
///   NSLocalizedString(@"description", @"")
class ObjCLocalizeStringLiteral : public Tweak {
public:
  const char *id() const override final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

  bool prepare(const Selection &Inputs) override;
  Expected<Tweak::Effect> apply(const Selection &Inputs) override;
  std::string title() const override;

private:
  const clang::ObjCStringLiteral *Str = nullptr;
};

REGISTER_TWEAK(ObjCLocalizeStringLiteral)

bool ObjCLocalizeStringLiteral::prepare(const Selection &Inputs) {
  const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
  if (!N)
    return false;
  // Allow the refactoring even if the user selected only the C string part
  // of the expression.
  if (N->ASTNode.get<StringLiteral>()) {
    if (N->Parent)
      N = N->Parent;
  }
  Str = dyn_cast_or_null<ObjCStringLiteral>(N->ASTNode.get<Stmt>());
  return Str;
}

Expected<Tweak::Effect>
ObjCLocalizeStringLiteral::apply(const Selection &Inputs) {
  auto *AST = Inputs.AST;
  auto &SM = AST->getSourceManager();
  const auto &TB = AST->getTokens();
  auto Toks = TB.spelledForExpanded(TB.expandedTokens(Str->getSourceRange()));
  if (!Toks || Toks->empty())
    return error("Failed to find tokens to replace.");
  // Insert `NSLocalizedString(` before the literal.
  auto Reps = tooling::Replacements(tooling::Replacement(
      SM, Toks->front().location(), 0, "NSLocalizedString("));
  // Insert `, @"")` after the literal.
  if (auto Err = Reps.add(
          tooling::Replacement(SM, Toks->back().endLocation(), 0, ", @\"\")")))
    return std::move(Err);
  return Effect::mainFileEdit(SM, std::move(Reps));
}

std::string ObjCLocalizeStringLiteral::title() const {
  return "Wrap in NSLocalizedString";
}

} // namespace
} // namespace clangd
} // namespace clang
