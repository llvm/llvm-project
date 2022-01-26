//===--- ExpandMacro.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Tweak.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <string>
namespace clang {
namespace clangd {
namespace {

/// Replaces a reference to a macro under the cursor with its expansion.
/// Before:
///   #define FOO(X) X+X
///   FOO(10*a)
///   ^^^
/// After:
///   #define FOO(X) X+X
///   10*a+10*a
class ExpandMacro : public Tweak {
public:
  const char *id() const override final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

  bool prepare(const Selection &Inputs) override;
  Expected<Tweak::Effect> apply(const Selection &Inputs) override;
  std::string title() const override;

private:
  syntax::TokenBuffer::Expansion Expansion;
  std::string MacroName;
};

REGISTER_TWEAK(ExpandMacro)

/// Finds a spelled token that the cursor is pointing at.
static const syntax::Token *
findTokenUnderCursor(const SourceManager &SM,
                     llvm::ArrayRef<syntax::Token> Spelled,
                     unsigned CursorOffset) {
  // Find the token that strats after the offset, then look at a previous one.
  auto *It = llvm::partition_point(Spelled, [&](const syntax::Token &T) {
    assert(T.location().isFileID());
    return SM.getFileOffset(T.location()) <= CursorOffset;
  });
  if (It == Spelled.begin())
    return nullptr;
  // Check the token we found actually touches the cursor position.
  --It;
  return It->range(SM).touches(CursorOffset) ? It : nullptr;
}

static const syntax::Token *
findIdentifierUnderCursor(const syntax::TokenBuffer &Tokens,
                          SourceLocation Cursor) {
  assert(Cursor.isFileID());

  auto &SM = Tokens.sourceManager();
  auto Spelled = Tokens.spelledTokens(SM.getFileID(Cursor));

  auto *T = findTokenUnderCursor(SM, Spelled, SM.getFileOffset(Cursor));
  if (!T)
    return nullptr;
  if (T->kind() == tok::identifier)
    return T;
  // Also try the previous token when the cursor is at the boundary, e.g.
  //   FOO^()
  //   FOO^+
  if (T == Spelled.begin())
    return nullptr;
  --T;
  if (T->endLocation() != Cursor || T->kind() != tok::identifier)
    return nullptr;
  return T;
}

bool ExpandMacro::prepare(const Selection &Inputs) {
  // FIXME: we currently succeed on selection at the end of the token, e.g.
  //        'FOO[[ ]]BAR'. We should not trigger in that case.

  // Find a token under the cursor.
  auto *T = findIdentifierUnderCursor(Inputs.AST->getTokens(), Inputs.Cursor);
  // We are interested only in identifiers, other tokens can't be macro names.
  if (!T)
    return false;
  // If the identifier is a macro we will find the corresponding expansion.
  auto Expansion = Inputs.AST->getTokens().expansionStartingAt(T);
  if (!Expansion)
    return false;
  this->MacroName = std::string(T->text(Inputs.AST->getSourceManager()));
  this->Expansion = *Expansion;
  return true;
}

Expected<Tweak::Effect> ExpandMacro::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();

  std::string Replacement;
  for (const syntax::Token &T : Expansion.Expanded) {
    Replacement += T.text(SM);
    Replacement += " ";
  }
  if (!Replacement.empty()) {
    assert(Replacement.back() == ' ');
    Replacement.pop_back();
  }

  CharSourceRange MacroRange =
      CharSourceRange::getCharRange(Expansion.Spelled.front().location(),
                                    Expansion.Spelled.back().endLocation());

  tooling::Replacements Reps;
  llvm::cantFail(Reps.add(tooling::Replacement(SM, MacroRange, Replacement)));
  return Effect::mainFileEdit(SM, std::move(Reps));
}

std::string ExpandMacro::title() const {
  return std::string(llvm::formatv("Expand macro '{0}'", MacroName));
}

} // namespace
} // namespace clangd
} // namespace clang
