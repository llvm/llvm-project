//===--- RawStringLiteral.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ParsedAST.h"
#include "refactor/Tweak.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {
/// Converts a string literal to a raw string.
/// Before:
///   printf("\"a\"\nb");
///          ^^^^^^^^^
/// After:
///   printf(R"("a"
/// b)");
class RawStringLiteral : public Tweak {
public:
  const char *id() const override final;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override { return "Convert to raw string"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  const clang::StringLiteral *Str = nullptr;
};

REGISTER_TWEAK(RawStringLiteral)

static bool isNormalString(const StringLiteral &Str, SourceLocation Cursor,
                          SourceManager &SM) {
  // All chunks must be normal ASCII strings, not u8"..." etc.
  if (!Str.isOrdinary())
    return false;
  SourceLocation LastTokenBeforeCursor;
  for (auto I = Str.tokloc_begin(), E = Str.tokloc_end(); I != E; ++I) {
    if (I->isMacroID()) // No tokens in the string may be macro expansions.
      return false;
    if (SM.isBeforeInTranslationUnit(*I, Cursor) || *I == Cursor)
      LastTokenBeforeCursor = *I;
  }
  // Token we care about must be a normal "string": not raw, u8, etc.
  const char* Data = SM.getCharacterData(LastTokenBeforeCursor);
  return Data && *Data == '"';
}

static bool needsRaw(llvm::StringRef Content) {
  return Content.find_first_of("\"\n\t") != StringRef::npos;
}

static bool canBeRaw(llvm::StringRef Content) {
  for (char C : Content)
    if (!llvm::isPrint(C) && C != '\n' && C != '\t')
      return false;
  return !Content.contains(")\"");
}

bool RawStringLiteral::prepare(const Selection &Inputs) {
  const SelectionTree::Node *N = Inputs.ASTSelection.commonAncestor();
  if (!N)
    return false;
  Str = dyn_cast_or_null<StringLiteral>(N->ASTNode.get<Stmt>());
  return Str &&
         isNormalString(*Str, Inputs.Cursor, Inputs.AST->getSourceManager()) &&
         needsRaw(Str->getBytes()) && canBeRaw(Str->getBytes());
}

Expected<Tweak::Effect> RawStringLiteral::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();
  auto Reps = tooling::Replacements(
      tooling::Replacement(SM, Str, ("R\"(" + Str->getBytes() + ")\"").str(),
                           Inputs.AST->getLangOpts()));
  return Effect::mainFileEdit(SM, std::move(Reps));
}

} // namespace
} // namespace clangd
} // namespace clang
