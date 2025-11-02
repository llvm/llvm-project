//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseUsingCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include <string>

using namespace clang::ast_matchers;
namespace {

AST_MATCHER(clang::LinkageSpecDecl, isExternCLinkage) {
  return Node.getLanguage() == clang::LinkageSpecLanguageIDs::C;
}
} // namespace

namespace clang::tidy::modernize {

static constexpr llvm::StringLiteral ExternCDeclName = "extern-c-decl";
static constexpr llvm::StringLiteral ParentDeclName = "parent-decl";
static constexpr llvm::StringLiteral TagDeclName = "tag-decl";
static constexpr llvm::StringLiteral TypedefName = "typedef";
static constexpr llvm::StringLiteral DeclStmtName = "decl-stmt";

UseUsingCheck::UseUsingCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreMacros(Options.get("IgnoreMacros", true)),
      IgnoreExternC(Options.get("IgnoreExternC", false)) {}

void UseUsingCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "IgnoreExternC", IgnoreExternC);
}

void UseUsingCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      typedefDecl(
          unless(isInstantiated()),
          optionally(hasAncestor(
              linkageSpecDecl(isExternCLinkage()).bind(ExternCDeclName))),
          anyOf(hasParent(decl().bind(ParentDeclName)),
                hasParent(declStmt().bind(DeclStmtName))))
          .bind(TypedefName),
      this);

  // This matcher is used to find tag declarations in source code within
  // typedefs. They appear in the AST just *prior* to the typedefs.
  Finder->addMatcher(
      tagDecl(
          anyOf(allOf(unless(anyOf(isImplicit(),
                                   classTemplateSpecializationDecl())),
                      anyOf(hasParent(decl().bind(ParentDeclName)),
                            hasParent(declStmt().bind(DeclStmtName)))),
                // We want the parent of the ClassTemplateDecl, not the parent
                // of the specialization.
                classTemplateSpecializationDecl(hasAncestor(classTemplateDecl(
                    anyOf(hasParent(decl().bind(ParentDeclName)),
                          hasParent(declStmt().bind(DeclStmtName))))))))
          .bind(TagDeclName),
      this);
}

void UseUsingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ParentDecl = Result.Nodes.getNodeAs<Decl>(ParentDeclName);

  if (!ParentDecl) {
    const auto *ParentDeclStmt = Result.Nodes.getNodeAs<DeclStmt>(DeclStmtName);
    if (ParentDeclStmt) {
      if (ParentDeclStmt->isSingleDecl())
        ParentDecl = ParentDeclStmt->getSingleDecl();
      else
        ParentDecl =
            ParentDeclStmt->getDeclGroup().getDeclGroup()
                [ParentDeclStmt->getDeclGroup().getDeclGroup().size() - 1];
    }
  }

  if (!ParentDecl)
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LO = getLangOpts();

  // Match CXXRecordDecl only to store the range of the last non-implicit full
  // declaration, to later check whether it's within the typedef itself.
  const auto *MatchedTagDecl = Result.Nodes.getNodeAs<TagDecl>(TagDeclName);
  if (MatchedTagDecl) {
    // It is not sufficient to just track the last TagDecl that we've seen,
    // because if one struct or union is nested inside another, the last TagDecl
    // before the typedef will be the nested one (PR#50990). Therefore, we also
    // keep track of the parent declaration, so that we can look up the last
    // TagDecl that is a sibling of the typedef in the AST.
    if (MatchedTagDecl->isThisDeclarationADefinition())
      LastTagDeclRanges[ParentDecl] = MatchedTagDecl->getSourceRange();
    return;
  }

  const auto *MatchedDecl = Result.Nodes.getNodeAs<TypedefDecl>(TypedefName);
  if (MatchedDecl->getLocation().isInvalid())
    return;

  const auto *ExternCDecl =
      Result.Nodes.getNodeAs<LinkageSpecDecl>(ExternCDeclName);
  if (ExternCDecl && IgnoreExternC)
    return;

  SourceLocation StartLoc = MatchedDecl->getBeginLoc();

  if (StartLoc.isMacroID() && IgnoreMacros)
    return;

  static constexpr llvm::StringLiteral UseUsingWarning =
      "use 'using' instead of 'typedef'";

  // Warn at StartLoc but do not fix if there is macro or array.
  if (MatchedDecl->getUnderlyingType()->isArrayType() || StartLoc.isMacroID()) {
    diag(StartLoc, UseUsingWarning);
    return;
  }

  const TypeLoc TL = MatchedDecl->getTypeSourceInfo()->getTypeLoc();

  auto [Type, QualifierStr] = [MatchedDecl, this, &TL, &SM,
                               &LO]() -> std::pair<std::string, std::string> {
    SourceRange TypeRange = TL.getSourceRange();

    // Function pointer case, get the left and right side of the identifier
    // without the identifier.
    if (TypeRange.fullyContains(MatchedDecl->getLocation())) {
      const auto RangeLeftOfIdentifier = CharSourceRange::getCharRange(
          TypeRange.getBegin(), MatchedDecl->getLocation());
      const auto RangeRightOfIdentifier = CharSourceRange::getCharRange(
          Lexer::getLocForEndOfToken(MatchedDecl->getLocation(), 0, SM, LO),
          Lexer::getLocForEndOfToken(TypeRange.getEnd(), 0, SM, LO));
      const std::string VerbatimType =
          (Lexer::getSourceText(RangeLeftOfIdentifier, SM, LO) +
           Lexer::getSourceText(RangeRightOfIdentifier, SM, LO))
              .str();
      return {VerbatimType, ""};
    }

    StringRef ExtraReference = "";
    if (MainTypeEndLoc.isValid() && TypeRange.fullyContains(MainTypeEndLoc)) {
      // Each type introduced in a typedef can specify being a reference or
      // pointer type separately, so we need to figure out if the new using-decl
      // needs to be to a reference or pointer as well.
      const SourceLocation Tok = utils::lexer::findPreviousAnyTokenKind(
          MatchedDecl->getLocation(), SM, LO, tok::TokenKind::star,
          tok::TokenKind::amp, tok::TokenKind::comma,
          tok::TokenKind::kw_typedef);

      ExtraReference = Lexer::getSourceText(
          CharSourceRange::getCharRange(Tok, Tok.getLocWithOffset(1)), SM, LO);

      if (ExtraReference != "*" && ExtraReference != "&")
        ExtraReference = "";

      TypeRange.setEnd(MainTypeEndLoc);
    }
    return {
        Lexer::getSourceText(CharSourceRange::getTokenRange(TypeRange), SM, LO)
            .str(),
        ExtraReference.str()};
  }();
  StringRef Name = MatchedDecl->getName();
  SourceRange ReplaceRange = MatchedDecl->getSourceRange();

  // typedefs with multiple comma-separated definitions produce multiple
  // consecutive TypedefDecl nodes whose SourceRanges overlap. Each range starts
  // at the "typedef" and then continues *across* previous definitions through
  // the end of the current TypedefDecl definition.
  // But also we need to check that the ranges belong to the same file because
  // different files may contain overlapping ranges.
  std::string Using = "using ";
  if (ReplaceRange.getBegin().isMacroID() ||
      (Result.SourceManager->getFileID(ReplaceRange.getBegin()) !=
       Result.SourceManager->getFileID(LastReplacementEnd)) ||
      (ReplaceRange.getBegin() >= LastReplacementEnd)) {
    // This is the first (and possibly the only) TypedefDecl in a typedef. Save
    // Type and Name in case we find subsequent TypedefDecl's in this typedef.
    FirstTypedefType = Type;
    FirstTypedefName = Name.str();
    MainTypeEndLoc = TL.getEndLoc();
  } else {
    // This is additional TypedefDecl in a comma-separated typedef declaration.
    // Start replacement *after* prior replacement and separate with semicolon.
    ReplaceRange.setBegin(LastReplacementEnd);
    Using = ";\nusing ";

    // If this additional TypedefDecl's Type starts with the first TypedefDecl's
    // type, make this using statement refer back to the first type, e.g. make
    // "typedef int Foo, *Foo_p;" -> "using Foo = int;\nusing Foo_p = Foo*;"
    if (Type == FirstTypedefType && !QualifierStr.empty())
      Type = FirstTypedefName;
  }

  if (!ReplaceRange.getEnd().isMacroID()) {
    const SourceLocation::IntTy Offset =
        MatchedDecl->getFunctionType() ? 0 : Name.size();
    LastReplacementEnd = ReplaceRange.getEnd().getLocWithOffset(Offset);
  }

  auto Diag = diag(ReplaceRange.getBegin(), UseUsingWarning);

  // If typedef contains a full tag declaration, extract its full text.
  auto LastTagDeclRange = LastTagDeclRanges.find(ParentDecl);
  if (LastTagDeclRange != LastTagDeclRanges.end() &&
      LastTagDeclRange->second.isValid() &&
      ReplaceRange.fullyContains(LastTagDeclRange->second)) {
    Type = std::string(Lexer::getSourceText(
        CharSourceRange::getTokenRange(LastTagDeclRange->second), SM, LO));
    if (Type.empty())
      return;
  }

  std::string Replacement = (Using + Name + " = " + Type + QualifierStr).str();
  Diag << FixItHint::CreateReplacement(ReplaceRange, Replacement);
}
} // namespace clang::tidy::modernize
