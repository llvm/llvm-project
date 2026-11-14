//===--- AddUsingReplaceAll.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Config.h"
#include "FindTarget.h"
#include "QualifiedNameRewrite.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <tuple>
#include <utility>

namespace clang {
namespace clangd {

/// Tweak for removing the full namespace qualifier under the cursor on
/// DeclRefExpr and types and adding the "using" statement instead. This tweak
/// replaces all occurrences of the qualified symbol in the file, not just the
/// one under the cursor. However, it also requires more preparation and is more
/// expensive to compute, so it is hidden behind a separate action from
/// AddUsing, which only removes the qualifier under the cursor.
class AddUsingReplaceAll : public Tweak {
public:
  const char *id() const override;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  NestedNameSpecifierLoc QualifierToRemove;
  std::string QualifierToSpell;
  llvm::StringRef SpelledQualifier;
  llvm::StringRef SpelledName;
  // The selected declaration must stay after the inserted using-declaration.
  SourceLocation MustInsertAfterLoc;
  // Canonical declarations whose explicit qualified references will be
  // rewritten in the main file.
  llvm::DenseSet<const NamedDecl *> RewriteTargets;
};
REGISTER_TWEAK(AddUsingReplaceAll)

std::string AddUsingReplaceAll::title() const {
  return std::string(llvm::formatv("Add using-declaration for {0} and replace "
                                   "all qualified references in this file",
                                   SpelledName));
}

struct InsertionPointData {
  SourceLocation Loc;
  std::string Suffix;
  bool AlwaysFullyQualify = false;
};

static llvm::Expected<InsertionPointData>
findInsertionPoint(const Tweak::Selection &Inputs,
                   const NestedNameSpecifierLoc &QualifierToRemove,
                   const llvm::StringRef Name,
                   const SourceLocation MustInsertAfterLoc) {
  auto &SM = Inputs.AST->getSourceManager();

  const auto *TargetNamespace =
      getNamespaceFromQualifier(QualifierToRemove.getNestedNameSpecifier());
  if (!TargetNamespace)
    return error("Qualifier is not a valid namespace qualifier");
  const auto *CanonicalTargetNS = TargetNamespace->getCanonicalDecl();

  auto IsValidPoint = [&](const SourceLocation Loc) {
    return MustInsertAfterLoc.isInvalid() ||
           SM.isBeforeInTranslationUnit(MustInsertAfterLoc, Loc);
  };

  SourceLocation LastUsingLoc;
  bool AlwaysFullyQualify = true;
  bool AlreadyExists = false;

  auto ProcessDecl = [&](const Decl *D) {
    const auto *UD = dyn_cast<UsingDecl>(D);
    if (!UD)
      return;
    SourceLocation UsingLoc = UD->getUsingLoc();
    if (SM.getFileID(UsingLoc) != SM.getMainFileID())
      return;

    if (!UD->getQualifier().isFullyQualified())
      AlwaysFullyQualify = false;

    if (const auto *Namespace = getNamespaceFromQualifier(UD->getQualifier())) {
      if (Namespace->getCanonicalDecl() == CanonicalTargetNS &&
          UD->getName() == Name) {
        AlreadyExists = true;
        return;
      }
    }

    if (LastUsingLoc.isInvalid() ||
        SM.isBeforeInTranslationUnit(LastUsingLoc, UsingLoc))
      LastUsingLoc = UsingLoc;
  };

  // Walk enclosing file/namespace contexts (skipping functions and classes) and
  // use getLocalTopLevelDecls() at translation-unit scope to avoid traversing
  // declarations from included headers.
  for (const DeclContext *Ctx =
           &Inputs.ASTSelection.commonAncestor()->getDeclContext();
       Ctx; Ctx = Ctx->getLexicalParent()) {
    if (!Ctx->isFileContext())
      continue;
    if (isa<TranslationUnitDecl>(Ctx)) {
      for (const Decl *D : Inputs.AST->getLocalTopLevelDecls()) {
        ProcessDecl(D);
        if (AlreadyExists)
          return InsertionPointData();
      }
    } else {
      for (const Decl *D : Ctx->decls()) {
        ProcessDecl(D);
        if (AlreadyExists)
          return InsertionPointData();
      }
    }
  }

  if (LastUsingLoc.isValid() && IsValidPoint(LastUsingLoc)) {
    InsertionPointData Out;
    Out.Loc = LastUsingLoc;
    Out.AlwaysFullyQualify = AlwaysFullyQualify;
    return Out;
  }

  const DeclContext *ParentDeclCtx =
      &Inputs.ASTSelection.commonAncestor()->getDeclContext();
  while (ParentDeclCtx && !ParentDeclCtx->isFileContext())
    ParentDeclCtx = ParentDeclCtx->getLexicalParent();

  if (auto *ND = llvm::dyn_cast_or_null<NamespaceDecl>(ParentDeclCtx)) {
    auto Toks = Inputs.AST->getTokens().expandedTokens(ND->getSourceRange());
    const auto *Tok = llvm::find_if(Toks, [](const syntax::Token &Tok) {
      return Tok.kind() == tok::l_brace;
    });
    if (Tok == Toks.end() || Tok->endLocation().isInvalid())
      return error("Namespace with no {{");
    if (!Tok->endLocation().isMacroID() && IsValidPoint(Tok->endLocation())) {
      InsertionPointData Out;
      Out.Loc = Tok->endLocation();
      Out.Suffix = "\n";
      return Out;
    }
  }

  auto TLDs = Inputs.AST->getLocalTopLevelDecls();
  for (const auto &TLD : TLDs) {
    if (!IsValidPoint(TLD->getBeginLoc()))
      continue;
    InsertionPointData Out;
    Out.Loc = SM.getExpansionLoc(TLD->getBeginLoc());
    Out.Suffix = "\n\n";
    return Out;
  }
  return error("Cannot find place to insert \"using\"");
}

static bool isNamespaceForbidden(const Tweak::Selection &Inputs,
                                 NestedNameSpecifier Namespace) {
  const auto *NS =
      dyn_cast<NamespaceDecl>(Namespace.getAsNamespaceAndPrefix().Namespace);
  if (!NS)
    return true;
  std::string NamespaceStr = printNamespaceScope(*NS);

  for (StringRef Banned : Config::current().Style.FullyQualifiedNamespaces) {
    StringRef PrefixMatch = NamespaceStr;
    if (PrefixMatch.consume_front(Banned) && PrefixMatch.consume_front("::"))
      return true;
  }

  return false;
}

bool AddUsingReplaceAll::prepare(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();

  // Clear any state from a previous invocation before analyzing the new
  // selection.
  QualifierToRemove = NestedNameSpecifierLoc();
  QualifierToSpell.clear();
  SpelledQualifier = llvm::StringRef();
  SpelledName = llvm::StringRef();
  MustInsertAfterLoc = SourceLocation();
  RewriteTargets.clear();

  if (isHeaderFile(SM.getFileEntryRefForID(SM.getMainFileID())->getName(),
                   Inputs.AST->getLangOpts()))
    return false;

  const auto &Prepared = prepareQualifiedSelection(
      Inputs, QualifiedRewriteClient::AddUsingReplaceAll);
  if (!Prepared.HasQualifiedRef ||
      isNamespaceForbidden(Inputs,
                           Prepared.QualifierToRemove.getNestedNameSpecifier()))
    return false;

  QualifierToRemove = Prepared.QualifierToRemove;
  QualifierToSpell = Prepared.QualifierToSpell;
  SpelledQualifier = Prepared.SpelledQualifier;
  SpelledName = Prepared.SpelledName;
  MustInsertAfterLoc = Prepared.MustInsertAfterLoc;
  RewriteTargets = Prepared.Targets;
  return true;
}

Expected<Tweak::Effect> AddUsingReplaceAll::apply(const Selection &Inputs) {
  auto &SM = Inputs.AST->getSourceManager();

  // Fail loudly if apply() is reached without the state produced by prepare().
  const auto *TargetNamespace =
      getNamespaceFromQualifier(QualifierToRemove.getNestedNameSpecifier());
  if (!TargetNamespace)
    return error("Invalid namespace qualifier in prepared state");
  if (SpelledName.empty() || QualifierToSpell.empty() || RewriteTargets.empty())
    return error("Incomplete prepared state for AddUsingReplaceAll");

  tooling::Replacements Repls;

  // Find a legal insertion point based on nearby using-declarations, namespace
  // scope, or the start of the file.
  auto InsertionPoint = findInsertionPoint(Inputs, QualifierToRemove,
                                           SpelledName, MustInsertAfterLoc);
  if (!InsertionPoint) {
    std::string Msg = llvm::toString(InsertionPoint.takeError());
    elog("AddUsingReplaceAll: {0}", Msg);
    return error(Msg);
  }

  if (InsertionPoint->Loc.isValid()) {
    // Emit the new using-declaration with any extra qualification required by
    // existing local style.
    std::string UsingText;
    llvm::raw_string_ostream UsingTextStream(UsingText);
    UsingTextStream << "using ";
    if (InsertionPoint->AlwaysFullyQualify &&
        !QualifierToRemove.getNestedNameSpecifier().isFullyQualified())
      UsingTextStream << "::";
    UsingTextStream << QualifierToSpell << SpelledName << ";"
                    << InsertionPoint->Suffix;

    assert(SM.getFileID(InsertionPoint->Loc) == SM.getMainFileID());
    if (auto Err = Repls.add(tooling::Replacement(SM, InsertionPoint->Loc, 0,
                                                  UsingTextStream.str())))
      return std::move(Err);
  }

  bool HasReplacementError = false;
  std::string ReplacementError;
  // Walk every top-level declaration in the file and rewrite only the
  // explicit references that resolve to one of the prepared targets.
  forEachQualifiedReferenceToRewrite(
      Inputs, RewriteTargets, [&](const QualifiedReference &Ref) {
        if (HasReplacementError)
          return;

        unsigned BeginOffset = SM.getFileOffset(Ref.QualifierLoc);
        unsigned EndOffset = SM.getFileOffset(Ref.NameLoc);
        if (BeginOffset >= EndOffset)
          return;

        // Drop the namespace qualifier, leaving the referenced name in
        // place.
        if (auto Err = Repls.add(tooling::Replacement(
                SM, Ref.QualifierLoc, EndOffset - BeginOffset, ""))) {
          ReplacementError = llvm::toString(std::move(Err));
          elog("AddUsingReplaceAll: {0}", ReplacementError);
          HasReplacementError = true;
          return;
        }
      });

  if (HasReplacementError)
    return error(ReplacementError);

  return Effect::mainFileEdit(SM, std::move(Repls));
}

} // namespace clangd
} // namespace clang
