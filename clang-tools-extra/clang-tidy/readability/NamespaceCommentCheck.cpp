//===--- NamespaceCommentCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NamespaceCommentCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

NamespaceCommentCheck::NamespaceCommentCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      NamespaceCommentPattern("^/[/*] *(end (of )?)? *(anonymous|unnamed)? *"
                              "namespace( +([a-zA-Z0-9_:]+))?\\.? *(\\*/)?$",
                              llvm::Regex::IgnoreCase),
      ShortNamespaceLines(Options.get("ShortNamespaceLines", 1u)),
      SpacesBeforeComments(Options.get("SpacesBeforeComments", 1u)) {}

void NamespaceCommentCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ShortNamespaceLines", ShortNamespaceLines);
  Options.store(Opts, "SpacesBeforeComments", SpacesBeforeComments);
}

void NamespaceCommentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(namespaceDecl().bind("namespace"), this);
}

static bool locationsInSameFile(const SourceManager &Sources,
                                SourceLocation Loc1, SourceLocation Loc2) {
  return Loc1.isFileID() && Loc2.isFileID() &&
         Sources.getFileID(Loc1) == Sources.getFileID(Loc2);
}

static std::optional<std::string>
getNamespaceNameAsWritten(SourceLocation &Loc, const SourceManager &Sources,
                          const LangOptions &LangOpts) {
  // Loc should be at the begin of the namespace decl (usually, `namespace`
  // token). We skip the first token right away, but in case of `inline
  // namespace` or `namespace a::inline b` we can see both `inline` and
  // `namespace` keywords, which we just ignore. Nested parens/squares before
  // the opening brace can result from attributes.
  std::string Result;
  int Nesting = 0;
  while (std::optional<Token> T = utils::lexer::findNextTokenSkippingComments(
             Loc, Sources, LangOpts)) {
    Loc = T->getLocation();
    if (T->is(tok::l_brace))
      break;

    if (T->isOneOf(tok::l_square, tok::l_paren)) {
      ++Nesting;
    } else if (T->isOneOf(tok::r_square, tok::r_paren)) {
      --Nesting;
    } else if (Nesting == 0) {
      if (T->is(tok::raw_identifier)) {
        StringRef ID = T->getRawIdentifier();
        if (ID != "namespace" && ID != "inline")
          Result.append(std::string(ID));
      } else if (T->is(tok::coloncolon)) {
        Result.append("::");
      } else { // Any other kind of token is unexpected here.
        return std::nullopt;
      }
    }
  }
  return Result;
}

void NamespaceCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ND = Result.Nodes.getNodeAs<NamespaceDecl>("namespace");
  const SourceManager &Sources = *Result.SourceManager;

  // Ignore namespaces inside macros and namespaces split across files.
  if (ND->getBeginLoc().isMacroID() ||
      !locationsInSameFile(Sources, ND->getBeginLoc(), ND->getRBraceLoc()))
    return;

  // Don't require closing comments for namespaces spanning less than certain
  // number of lines.
  unsigned StartLine = Sources.getSpellingLineNumber(ND->getBeginLoc());
  unsigned EndLine = Sources.getSpellingLineNumber(ND->getRBraceLoc());
  if (EndLine - StartLine + 1 <= ShortNamespaceLines)
    return;

  // Find next token after the namespace closing brace.
  SourceLocation AfterRBrace = Lexer::getLocForEndOfToken(
      ND->getRBraceLoc(), /*Offset=*/0, Sources, getLangOpts());
  SourceLocation Loc = AfterRBrace;
  SourceLocation LBraceLoc = ND->getBeginLoc();

  // Currently for nested namespace (n1::n2::...) the AST matcher will match foo
  // then bar instead of a single match. So if we got a nested namespace we have
  // to skip the next ones.
  for (const auto &EndOfNameLocation : Ends) {
    if (Sources.isBeforeInTranslationUnit(ND->getLocation(), EndOfNameLocation))
      return;
  }

  std::optional<std::string> NamespaceNameAsWritten =
      getNamespaceNameAsWritten(LBraceLoc, Sources, getLangOpts());
  if (!NamespaceNameAsWritten)
    return;

  if (NamespaceNameAsWritten->empty() != ND->isAnonymousNamespace()) {
    // Apparently, we didn't find the correct namespace name. Give up.
    return;
  }

  Ends.push_back(LBraceLoc);

  Token Tok;
  // Skip whitespace until we find the next token.
  while (Lexer::getRawToken(Loc, Tok, Sources, getLangOpts()) ||
         Tok.is(tok::semi)) {
    Loc = Loc.getLocWithOffset(1);
  }

  if (!locationsInSameFile(Sources, ND->getRBraceLoc(), Loc))
    return;

  bool NextTokenIsOnSameLine = Sources.getSpellingLineNumber(Loc) == EndLine;
  // If we insert a line comment before the token in the same line, we need
  // to insert a line break.
  bool NeedLineBreak = NextTokenIsOnSameLine && Tok.isNot(tok::eof);

  SourceRange OldCommentRange(AfterRBrace, AfterRBrace);
  std::string Message = "%0 not terminated with a closing comment";

  // Try to find existing namespace closing comment on the same line.
  if (Tok.is(tok::comment) && NextTokenIsOnSameLine) {
    StringRef Comment(Sources.getCharacterData(Loc), Tok.getLength());
    SmallVector<StringRef, 7> Groups;
    if (NamespaceCommentPattern.match(Comment, &Groups)) {
      StringRef NamespaceNameInComment = Groups.size() > 5 ? Groups[5] : "";
      StringRef Anonymous = Groups.size() > 3 ? Groups[3] : "";

      if ((ND->isAnonymousNamespace() && NamespaceNameInComment.empty()) ||
          (*NamespaceNameAsWritten == NamespaceNameInComment &&
           Anonymous.empty())) {
        // Check if the namespace in the comment is the same.
        // FIXME: Maybe we need a strict mode, where we always fix namespace
        // comments with different format.
        return;
      }

      // Otherwise we need to fix the comment.
      NeedLineBreak = Comment.startswith("/*");
      OldCommentRange =
          SourceRange(AfterRBrace, Loc.getLocWithOffset(Tok.getLength()));
      Message =
          (llvm::Twine(
               "%0 ends with a comment that refers to a wrong namespace '") +
           NamespaceNameInComment + "'")
              .str();
    } else if (Comment.startswith("//")) {
      // Assume that this is an unrecognized form of a namespace closing line
      // comment. Replace it.
      NeedLineBreak = false;
      OldCommentRange =
          SourceRange(AfterRBrace, Loc.getLocWithOffset(Tok.getLength()));
      Message = "%0 ends with an unrecognized comment";
    }
    // If it's a block comment, just move it to the next line, as it can be
    // multi-line or there may be other tokens behind it.
  }

  std::string NamespaceNameForDiag =
      ND->isAnonymousNamespace() ? "anonymous namespace"
                                 : ("namespace '" + *NamespaceNameAsWritten + "'");

  std::string Fix(SpacesBeforeComments, ' ');
  Fix.append("// namespace");
  if (!ND->isAnonymousNamespace())
    Fix.append(" ").append(*NamespaceNameAsWritten);
  if (NeedLineBreak)
    Fix.append("\n");

  // Place diagnostic at an old comment, or closing brace if we did not have it.
  SourceLocation DiagLoc =
      OldCommentRange.getBegin() != OldCommentRange.getEnd()
          ? OldCommentRange.getBegin()
          : ND->getRBraceLoc();

  diag(DiagLoc, Message) << NamespaceNameForDiag
                         << FixItHint::CreateReplacement(
                                CharSourceRange::getCharRange(OldCommentRange),
                                Fix);
  diag(ND->getLocation(), "%0 starts here", DiagnosticIDs::Note)
      << NamespaceNameForDiag;
}

} // namespace clang::tidy::readability
