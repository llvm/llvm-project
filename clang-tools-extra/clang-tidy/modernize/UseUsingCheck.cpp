//===--- UseUsingCheck.cpp - clang-tidy------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseUsingCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Lex/Lexer.h"

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
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)),
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

  // Match CXXRecordDecl only to store the range of the last non-implicit full
  // declaration, to later check whether it's within the typdef itself.
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

  static const char *UseUsingWarning = "use 'using' instead of 'typedef'";

  // Warn at StartLoc but do not fix if there is macro or array.
  if (MatchedDecl->getUnderlyingType()->isArrayType() || StartLoc.isMacroID()) {
    diag(StartLoc, UseUsingWarning);
    return;
  }

  PrintingPolicy PrintPolicy(getLangOpts());
  PrintPolicy.SuppressScope = true;
  PrintPolicy.ConstantArraySizeAsWritten = true;
  PrintPolicy.UseVoidForZeroParams = false;
  PrintPolicy.PrintInjectedClassNameWithArguments = false;

  std::string Type = MatchedDecl->getUnderlyingType().getAsString(PrintPolicy);
  std::string Name = MatchedDecl->getNameAsString();
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
    FirstTypedefName = Name;
  } else {
    // This is additional TypedefDecl in a comma-separated typedef declaration.
    // Start replacement *after* prior replacement and separate with semicolon.
    ReplaceRange.setBegin(LastReplacementEnd);
    Using = ";\nusing ";

    // If this additional TypedefDecl's Type starts with the first TypedefDecl's
    // type, make this using statement refer back to the first type, e.g. make
    // "typedef int Foo, *Foo_p;" -> "using Foo = int;\nusing Foo_p = Foo*;"
    if (Type.size() > FirstTypedefType.size() &&
        Type.substr(0, FirstTypedefType.size()) == FirstTypedefType)
      Type = FirstTypedefName + Type.substr(FirstTypedefType.size() + 1);
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
        CharSourceRange::getTokenRange(LastTagDeclRange->second),
        *Result.SourceManager, getLangOpts()));
    if (Type.empty())
      return;
  }

  std::string Replacement = Using + Name + " = " + Type;
  Diag << FixItHint::CreateReplacement(ReplaceRange, Replacement);
}
} // namespace clang::tidy::modernize
