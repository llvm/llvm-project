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

namespace lexer = clang::tidy::utils::lexer;

namespace {
struct RangeTextInfo {
  std::string Text;
  lexer::TokenRangeInfo Tokens;
};
} // namespace

static bool hasNonWhitespace(llvm::StringRef Text) {
  return Text.find_first_not_of(" \t\n\r\f\v") != llvm::StringRef::npos;
}

static RangeTextInfo getRangeTextInfo(clang::SourceLocation Begin,
                                      clang::SourceLocation End,
                                      const clang::SourceManager &SM,
                                      const clang::LangOptions &LangOpts) {
  RangeTextInfo Info;
  if (!Begin.isValid() || !End.isValid() || Begin.isMacroID() ||
      End.isMacroID())
    return Info;

  const clang::CharSourceRange Range =
      clang::CharSourceRange::getCharRange(Begin, End);
  Info.Text = lexer::getSourceText(Range, SM, LangOpts);
  Info.Tokens = lexer::analyzeTokenRange(Range, SM, LangOpts);
  return Info;
}

static std::string getFunctionPointerTypeText(clang::SourceRange TypeRange,
                                              clang::SourceLocation NameLoc,
                                              const clang::SourceManager &SM,
                                              const clang::LangOptions &LO) {
  clang::SourceLocation StartLoc = NameLoc;
  clang::SourceLocation EndLoc = NameLoc;

  while (true) {
    const std::optional<clang::Token> Prev =
        lexer::getPreviousToken(StartLoc, SM, LO);
    const std::optional<clang::Token> Next =
        lexer::findNextTokenSkippingComments(EndLoc, SM, LO);
    if (!Prev || Prev->isNot(clang::tok::l_paren) || !Next ||
        Next->isNot(clang::tok::r_paren))
      break;

    StartLoc = Prev->getLocation();
    EndLoc = Next->getLocation();
  }

  const clang::CharSourceRange RangeLeftOfIdentifier =
      clang::CharSourceRange::getCharRange(TypeRange.getBegin(), StartLoc);
  const clang::CharSourceRange RangeRightOfIdentifier =
      clang::CharSourceRange::getCharRange(
          clang::Lexer::getLocForEndOfToken(EndLoc, 0, SM, LO),
          clang::Lexer::getLocForEndOfToken(TypeRange.getEnd(), 0, SM, LO));
  return lexer::getSourceText(RangeLeftOfIdentifier, SM, LO) +
         lexer::getSourceText(RangeRightOfIdentifier, SM, LO);
}

static RangeTextInfo getLeadingTextInfo(bool IsFirstTypedefInGroup,
                                        clang::SourceRange ReplaceRange,
                                        clang::SourceRange TypeRange,
                                        const clang::SourceManager &SM,
                                        const clang::LangOptions &LO) {
  RangeTextInfo Info;
  if (!IsFirstTypedefInGroup)
    return Info;

  const clang::SourceLocation TypedefEnd =
      clang::Lexer::getLocForEndOfToken(ReplaceRange.getBegin(), 0, SM, LO);
  Info = getRangeTextInfo(TypedefEnd, TypeRange.getBegin(), SM, LO);
  // Keep leading trivia only when it actually contains comments. This avoids
  // shifting plain whitespace from between 'typedef' and the type into the
  // replacement, preserving formatting for un-commented declarations.
  if (!Info.Tokens.HasComment)
    Info.Text.clear();
  return Info;
}

static RangeTextInfo getSuffixTextInfo(bool FunctionPointerCase,
                                       bool IsFirstTypedefInGroup,
                                       clang::SourceLocation PrevReplacementEnd,
                                       clang::SourceRange TypeRange,
                                       clang::SourceLocation NameLoc,
                                       const clang::SourceManager &SM,
                                       const clang::LangOptions &LO) {
  RangeTextInfo Info;
  if (FunctionPointerCase)
    return Info;

  // Capture the raw text between type and name to preserve trailing comments,
  // including multi-line // blocks, without re-lexing individual comment
  // tokens.
  if (IsFirstTypedefInGroup) {
    const clang::SourceLocation AfterType =
        clang::Lexer::getLocForEndOfToken(TypeRange.getEnd(), 0, SM, LO);
    return getRangeTextInfo(AfterType, NameLoc, SM, LO);
  }

  if (!PrevReplacementEnd.isValid() || PrevReplacementEnd.isMacroID())
    return Info;

  clang::SourceLocation AfterComma = PrevReplacementEnd;
  if (const std::optional<clang::Token> NextTok =
          lexer::findNextTokenSkippingComments(AfterComma, SM, LO)) {
    if (NextTok->is(clang::tok::comma)) {
      AfterComma =
          clang::Lexer::getLocForEndOfToken(NextTok->getLocation(), 0, SM, LO);
    }
  }
  return getRangeTextInfo(AfterComma, NameLoc, SM, LO);
}

static void stripLeadingComma(RangeTextInfo &Info) {
  if (Info.Text.empty())
    return;
  // Overlapping ranges in multi-declarator typedefs can leave a leading comma
  // in the captured suffix. Drop it so the replacement doesn't reintroduce it.
  const size_t NonWs = Info.Text.find_first_not_of(" \t\n\r\f\v");
  if (NonWs != std::string::npos && Info.Text[NonWs] == ',')
    Info.Text.erase(0, NonWs + 1);
}

static constexpr StringRef ExternCDeclName = "extern-c-decl";
static constexpr StringRef ParentDeclName = "parent-decl";
static constexpr StringRef TagDeclName = "tag-decl";
static constexpr StringRef TypedefName = "typedef";
static constexpr StringRef DeclStmtName = "decl-stmt";

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

  const SourceLocation StartLoc = MatchedDecl->getBeginLoc();

  if (StartLoc.isMacroID() && IgnoreMacros)
    return;

  static constexpr StringRef UseUsingWarning =
      "use 'using' instead of 'typedef'";

  // Warn at StartLoc but do not fix if there is macro or array.
  if (MatchedDecl->getUnderlyingType()->isArrayType() || StartLoc.isMacroID()) {
    diag(StartLoc, UseUsingWarning);
    return;
  }

  const TypeLoc TL = MatchedDecl->getTypeSourceInfo()->getTypeLoc();

  struct TypeInfo {
    SourceRange Range;
    bool FunctionPointerCase = false;
    std::string Type;
    std::string Qualifier;
  };

  const TypeInfo TI = [&] {
    TypeInfo Info;
    Info.Range = TL.getSourceRange();

    // Function pointer case, get the left and right side of the identifier
    // without the identifier.
    if (Info.Range.fullyContains(MatchedDecl->getLocation())) {
      Info.FunctionPointerCase = true;
      Info.Type = getFunctionPointerTypeText(
          Info.Range, MatchedDecl->getLocation(), SM, LO);
      return Info;
    }

    std::string ExtraReference;
    if (MainTypeEndLoc.isValid() && Info.Range.fullyContains(MainTypeEndLoc)) {
      // Each type introduced in a typedef can specify being a reference or
      // pointer type separately, so we need to figure out if the new using-decl
      // needs to be to a reference or pointer as well.
      const SourceLocation Tok = lexer::findPreviousAnyTokenKind(
          MatchedDecl->getLocation(), SM, LO, tok::TokenKind::star,
          tok::TokenKind::amp, tok::TokenKind::comma,
          tok::TokenKind::kw_typedef);

      ExtraReference = lexer::getSourceText(
          CharSourceRange::getCharRange(Tok, Tok.getLocWithOffset(1)), SM, LO);

      if (ExtraReference != "*" && ExtraReference != "&")
        ExtraReference = "";

      Info.Range.setEnd(MainTypeEndLoc);
    }

    Info.Type = lexer::getSourceText(CharSourceRange::getTokenRange(Info.Range),
                                     SM, LO);
    Info.Qualifier = ExtraReference;
    return Info;
  }();

  const SourceRange TypeRange = TI.Range;
  const bool FunctionPointerCase = TI.FunctionPointerCase;
  std::string Type = TI.Type;
  const std::string QualifierStr = TI.Qualifier;
  const StringRef Name = MatchedDecl->getName();
  const SourceLocation NameLoc = MatchedDecl->getLocation();
  SourceRange ReplaceRange = MatchedDecl->getSourceRange();
  const SourceLocation PrevReplacementEnd = LastReplacementEnd;

  // typedefs with multiple comma-separated definitions produce multiple
  // consecutive TypedefDecl nodes whose SourceRanges overlap. Each range starts
  // at the "typedef" and then continues *across* previous definitions through
  // the end of the current TypedefDecl definition.
  // But also we need to check that the ranges belong to the same file because
  // different files may contain overlapping ranges.
  std::string Using = "using ";
  const bool IsFirstTypedefInGroup =
      ReplaceRange.getBegin().isMacroID() ||
      (Result.SourceManager->getFileID(ReplaceRange.getBegin()) !=
       Result.SourceManager->getFileID(LastReplacementEnd)) ||
      (ReplaceRange.getBegin() >= LastReplacementEnd);

  if (IsFirstTypedefInGroup) {
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

  const RangeTextInfo LeadingTextInfo = getLeadingTextInfo(
      IsFirstTypedefInGroup, ReplaceRange, TypeRange, SM, LO);
  RangeTextInfo SuffixTextInfo =
      getSuffixTextInfo(FunctionPointerCase, IsFirstTypedefInGroup,
                        PrevReplacementEnd, TypeRange, NameLoc, SM, LO);
  if (!IsFirstTypedefInGroup)
    stripLeadingComma(SuffixTextInfo);

  const bool SuffixHasComment = SuffixTextInfo.Tokens.HasComment;
  std::string SuffixText;
  if (SuffixHasComment) {
    SuffixText = SuffixTextInfo.Text;
  } else if (QualifierStr.empty() && hasNonWhitespace(SuffixTextInfo.Text) &&
             SuffixTextInfo.Tokens.HasPointerOrRef &&
             !SuffixTextInfo.Tokens.HasIdentifier) {
    // Only keep non-comment suffix text when it's purely pointer/ref trivia.
    // This avoids accidentally pulling in keywords like 'typedef'.
    SuffixText = SuffixTextInfo.Text;
  }
  const std::string QualifierText = SuffixHasComment ? "" : QualifierStr;

  if (!ReplaceRange.getEnd().isMacroID()) {
    const SourceLocation::IntTy Offset = FunctionPointerCase ? 0 : Name.size();
    LastReplacementEnd = ReplaceRange.getEnd().getLocWithOffset(Offset);
  }

  auto Diag = diag(ReplaceRange.getBegin(), UseUsingWarning);

  // If typedef contains a full tag declaration, extract its full text.
  auto LastTagDeclRange = LastTagDeclRanges.find(ParentDecl);
  if (LastTagDeclRange != LastTagDeclRanges.end() &&
      LastTagDeclRange->second.isValid() &&
      ReplaceRange.fullyContains(LastTagDeclRange->second)) {
    Type = lexer::getSourceText(
        CharSourceRange::getTokenRange(LastTagDeclRange->second), SM, LO);
    if (Type.empty())
      return;
  }

  std::string TypeExpr =
      LeadingTextInfo.Text + Type + QualifierText + SuffixText;
  TypeExpr = StringRef(TypeExpr).rtrim(" \t").str();
  StringRef Assign = " = ";
  if (!TypeExpr.empty() &&
      (TypeExpr.front() == ' ' || TypeExpr.front() == '\t'))
    Assign = " =";
  const std::string Replacement = (Using + Name + Assign + TypeExpr).str();
  Diag << FixItHint::CreateReplacement(ReplaceRange, Replacement);
}
} // namespace clang::tidy::modernize
