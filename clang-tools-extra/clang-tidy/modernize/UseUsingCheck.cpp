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

namespace clang::tidy::modernize {

namespace lexer = clang::tidy::utils::lexer;

namespace {

AST_MATCHER(LinkageSpecDecl, isExternCLinkage) {
  return Node.getLanguage() == LinkageSpecLanguageIDs::C;
}

} // namespace

namespace {

struct TokenRangeInfo {
  bool HasComment = false;
  bool HasIdentifier = false;
  bool HasPointerOrRef = false;
};

struct RangeTextInfo {
  std::string Text;
  TokenRangeInfo Tokens;
};

} // namespace

static StringRef::size_type findFirstNonWhitespace(StringRef Text) {
  return Text.find_first_not_of(" \t\n\r\f\v");
}

static std::optional<std::string> getSourceText(CharSourceRange Range,
                                                const SourceManager &SM,
                                                const LangOptions &LangOpts) {
  if (Range.isInvalid())
    return std::nullopt;

  const CharSourceRange FileRange =
      Lexer::makeFileCharRange(Range, SM, LangOpts);
  if (FileRange.isInvalid())
    return std::nullopt;

  bool IsInvalid = false;
  const StringRef Text =
      Lexer::getSourceText(FileRange, SM, LangOpts, &IsInvalid);
  if (IsInvalid)
    return std::nullopt;
  return Text.str();
}

static TokenRangeInfo getTokenRangeInfo(CharSourceRange Range,
                                        const SourceManager &SM,
                                        const LangOptions &LangOpts) {
  TokenRangeInfo Info;
  if (Range.isInvalid())
    return Info;

  const CharSourceRange FileRange =
      Lexer::makeFileCharRange(Range, SM, LangOpts);
  if (FileRange.isInvalid())
    return Info;

  const auto [BeginFID, BeginOffset] =
      SM.getDecomposedLoc(FileRange.getBegin());
  const auto [EndFID, EndOffset] = SM.getDecomposedLoc(FileRange.getEnd());
  if (BeginFID != EndFID || BeginOffset > EndOffset)
    return Info;

  bool IsInvalid = false;
  const StringRef Buffer = SM.getBufferData(BeginFID, &IsInvalid);
  if (IsInvalid)
    return Info;

  const char *LexStart = Buffer.data() + BeginOffset;
  Lexer TheLexer(SM.getLocForStartOfFile(BeginFID), LangOpts, Buffer.begin(),
                 LexStart, Buffer.end());
  TheLexer.SetCommentRetentionState(true);

  while (true) {
    Token Tok;
    if (TheLexer.LexFromRawLexer(Tok))
      break;

    if (Tok.is(tok::eof) || Tok.getLocation() == FileRange.getEnd() ||
        SM.isBeforeInTranslationUnit(FileRange.getEnd(), Tok.getLocation()))
      break;

    if (Tok.is(tok::comment)) {
      Info.HasComment = true;
      continue;
    }

    if (Tok.isOneOf(tok::star, tok::amp))
      Info.HasPointerOrRef = true;

    if (tok::isAnyIdentifier(Tok.getKind()) ||
        Tok.isOneOf(tok::kw_typedef, tok::kw_struct, tok::kw_class,
                    tok::kw_union, tok::kw_enum, tok::kw_typename,
                    tok::kw_template)) {
      Info.HasIdentifier = true;
    }
  }

  return Info;
}

static RangeTextInfo getRangeTextInfo(SourceLocation Begin, SourceLocation End,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts) {
  if (!Begin.isValid() || !End.isValid() || Begin.isMacroID() ||
      End.isMacroID())
    return {};

  const CharSourceRange Range = CharSourceRange::getCharRange(Begin, End);
  if (std::optional<std::string> Text = getSourceText(Range, SM, LangOpts))
    return {*Text, getTokenRangeInfo(Range, SM, LangOpts)};
  return {};
}

static std::optional<std::string>
getFunctionPointerTypeText(SourceRange TypeRange, SourceLocation NameLoc,
                           const SourceManager &SM, const LangOptions &LO) {
  SourceLocation StartLoc = NameLoc;
  SourceLocation EndLoc = NameLoc;

  while (true) {
    const std::optional<Token> Prev = lexer::getPreviousToken(StartLoc, SM, LO);
    const std::optional<Token> Next =
        lexer::findNextTokenSkippingComments(EndLoc, SM, LO);
    if (!Prev || Prev->isNot(tok::l_paren) || !Next ||
        Next->isNot(tok::r_paren))
      break;

    StartLoc = Prev->getLocation();
    EndLoc = Next->getLocation();
  }

  const CharSourceRange RangeLeftOfIdentifier =
      CharSourceRange::getCharRange(TypeRange.getBegin(), StartLoc);
  const CharSourceRange RangeRightOfIdentifier = CharSourceRange::getCharRange(
      Lexer::getLocForEndOfToken(EndLoc, 0, SM, LO),
      Lexer::getLocForEndOfToken(TypeRange.getEnd(), 0, SM, LO));

  const std::optional<std::string> LeftText =
      getSourceText(RangeLeftOfIdentifier, SM, LO);
  if (!LeftText)
    return std::nullopt;

  const std::optional<std::string> RightText =
      getSourceText(RangeRightOfIdentifier, SM, LO);
  if (!RightText)
    return std::nullopt;

  return *LeftText + *RightText;
}

static RangeTextInfo getLeadingTextInfo(bool IsFirstTypedefInGroup,
                                        SourceRange ReplaceRange,
                                        SourceRange TypeRange,
                                        const SourceManager &SM,
                                        const LangOptions &LO) {
  if (!IsFirstTypedefInGroup)
    return {};

  const SourceLocation TypedefEnd =
      Lexer::getLocForEndOfToken(ReplaceRange.getBegin(), 0, SM, LO);
  RangeTextInfo Info =
      getRangeTextInfo(TypedefEnd, TypeRange.getBegin(), SM, LO);
  if (!Info.Tokens.HasComment)
    Info.Text.clear();
  return Info;
}

static RangeTextInfo
getSuffixTextInfo(bool FunctionPointerCase, bool IsFirstTypedefInGroup,
                  SourceLocation PrevReplacementEnd, SourceRange TypeRange,
                  SourceLocation NameLoc, const SourceManager &SM,
                  const LangOptions &LO) {
  if (FunctionPointerCase)
    return {};

  if (IsFirstTypedefInGroup) {
    const SourceLocation AfterType =
        Lexer::getLocForEndOfToken(TypeRange.getEnd(), 0, SM, LO);
    return getRangeTextInfo(AfterType, NameLoc, SM, LO);
  }

  if (!PrevReplacementEnd.isValid() || PrevReplacementEnd.isMacroID())
    return {};

  SourceLocation AfterComma = PrevReplacementEnd;
  if (const std::optional<Token> NextTok =
          lexer::findNextTokenSkippingComments(AfterComma, SM, LO)) {
    if (NextTok->is(tok::comma)) {
      AfterComma =
          Lexer::getLocForEndOfToken(NextTok->getLocation(), 0, SM, LO);
    }
  }

  return getRangeTextInfo(AfterComma, NameLoc, SM, LO);
}

static void stripLeadingComma(RangeTextInfo &Info) {
  const StringRef::size_type NonWs = findFirstNonWhitespace(Info.Text);
  if (NonWs != StringRef::npos && Info.Text[NonWs] == ',')
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
    bool Valid = false;
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
      if (std::optional<std::string> Type = getFunctionPointerTypeText(
              Info.Range, MatchedDecl->getLocation(), SM, LO)) {
        Info.Type = *Type;
        Info.Valid = true;
      }
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

      const std::optional<std::string> Reference = getSourceText(
          CharSourceRange::getCharRange(Tok, Tok.getLocWithOffset(1)), SM, LO);
      if (!Reference)
        return Info;
      ExtraReference = *Reference;

      if (ExtraReference != "*" && ExtraReference != "&")
        ExtraReference.clear();

      Info.Range.setEnd(MainTypeEndLoc);
    }

    if (std::optional<std::string> Type =
            getSourceText(CharSourceRange::getTokenRange(Info.Range), SM, LO)) {
      Info.Type = *Type;
      Info.Qualifier = ExtraReference;
      Info.Valid = true;
    }
    return Info;
  }();

  if (!TI.Valid) {
    diag(StartLoc, UseUsingWarning);
    return;
  }

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
  } else if (QualifierStr.empty() &&
             findFirstNonWhitespace(SuffixTextInfo.Text) != StringRef::npos &&
             SuffixTextInfo.Tokens.HasPointerOrRef &&
             !SuffixTextInfo.Tokens.HasIdentifier) {
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
    const std::optional<std::string> TagType = getSourceText(
        CharSourceRange::getTokenRange(LastTagDeclRange->second), SM, LO);
    if (!TagType)
      return;
    Type = *TagType;
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
