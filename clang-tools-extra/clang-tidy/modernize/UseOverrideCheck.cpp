//===--- UseOverrideCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseOverrideCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

void UseOverrideCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreDestructors", IgnoreDestructors);
}

void UseOverrideCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matcher for C++11.
  if (!getLangOpts().CPlusPlus11)
    return;

  if (IgnoreDestructors)
    Finder->addMatcher(
        cxxMethodDecl(isOverride(), unless(cxxDestructorDecl())).bind("method"),
        this);
  else
    Finder->addMatcher(cxxMethodDecl(isOverride()).bind("method"), this);
}

// Re-lex the tokens to get precise locations to insert 'override' and remove
// 'virtual'.
static SmallVector<Token, 16>
ParseTokens(CharSourceRange Range, const MatchFinder::MatchResult &Result) {
  const SourceManager &Sources = *Result.SourceManager;
  std::pair<FileID, unsigned> LocInfo =
      Sources.getDecomposedLoc(Range.getBegin());
  StringRef File = Sources.getBufferData(LocInfo.first);
  const char *TokenBegin = File.data() + LocInfo.second;
  Lexer RawLexer(Sources.getLocForStartOfFile(LocInfo.first),
                 Result.Context->getLangOpts(), File.begin(), TokenBegin,
                 File.end());
  SmallVector<Token, 16> Tokens;
  Token Tok;
  int NestedParens = 0;
  while (!RawLexer.LexFromRawLexer(Tok)) {
    if ((Tok.is(tok::semi) || Tok.is(tok::l_brace)) && NestedParens == 0)
      break;
    if (Sources.isBeforeInTranslationUnit(Range.getEnd(), Tok.getLocation()))
      break;
    if (Tok.is(tok::l_paren))
      ++NestedParens;
    else if (Tok.is(tok::r_paren))
      --NestedParens;
    if (Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = Result.Context->Idents.get(StringRef(
          Sources.getCharacterData(Tok.getLocation()), Tok.getLength()));
      Tok.setIdentifierInfo(&Info);
      Tok.setKind(Info.getTokenID());
    }
    Tokens.push_back(Tok);
  }
  return Tokens;
}

static StringRef GetText(const Token &Tok, const SourceManager &Sources) {
  return StringRef(Sources.getCharacterData(Tok.getLocation()),
                   Tok.getLength());
}

void UseOverrideCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Method = Result.Nodes.getNodeAs<FunctionDecl>("method");
  const SourceManager &Sources = *Result.SourceManager;

  assert(Method != nullptr);
  if (Method->getInstantiatedFromMemberFunction() != nullptr)
    Method = Method->getInstantiatedFromMemberFunction();

  if (Method->isImplicit() || Method->getLocation().isMacroID() ||
      Method->isOutOfLine())
    return;

  bool HasVirtual = Method->isVirtualAsWritten();
  bool HasOverride = Method->getAttr<OverrideAttr>();
  bool HasFinal = Method->getAttr<FinalAttr>();

  bool OnlyVirtualSpecified = HasVirtual && !HasOverride && !HasFinal;
  unsigned KeywordCount = HasVirtual + HasOverride + HasFinal;

  if (!OnlyVirtualSpecified && KeywordCount == 1)
    return; // Nothing to do.

  std::string Message;

  if (OnlyVirtualSpecified) {
    Message =
        "prefer using 'override' or (rarely) 'final' instead of 'virtual'";
  } else if (KeywordCount == 0) {
    Message = "annotate this function with 'override' or (rarely) 'final'";
  } else {
    StringRef Redundant =
        HasVirtual ? (HasOverride && HasFinal ? "'virtual' and 'override' are"
                                              : "'virtual' is")
                   : "'override' is";
    StringRef Correct = HasFinal ? "'final'" : "'override'";

    Message = (llvm::Twine(Redundant) +
               " redundant since the function is already declared " + Correct)
                  .str();
  }

  DiagnosticBuilder Diag = diag(Method->getLocation(), Message);

  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Method->getSourceRange()), Sources,
      getLangOpts());

  if (!FileRange.isValid())
    return;

  // FIXME: Instead of re-lexing and looking for specific macros such as
  // 'ABSTRACT', properly store the location of 'virtual' and '= 0' in each
  // FunctionDecl.
  SmallVector<Token, 16> Tokens = ParseTokens(FileRange, Result);

  // Add 'override' on inline declarations that don't already have it.
  if (!HasFinal && !HasOverride) {
    SourceLocation InsertLoc;
    StringRef ReplacementText = "override ";
    SourceLocation MethodLoc = Method->getLocation();

    for (Token T : Tokens) {
      if (T.is(tok::kw___attribute) &&
          !Sources.isBeforeInTranslationUnit(T.getLocation(), MethodLoc)) {
        InsertLoc = T.getLocation();
        break;
      }
    }

    if (Method->hasAttrs()) {
      for (const clang::Attr *A : Method->getAttrs()) {
        if (!A->isImplicit() && !A->isInherited()) {
          SourceLocation Loc =
              Sources.getExpansionLoc(A->getRange().getBegin());
          if ((!InsertLoc.isValid() ||
               Sources.isBeforeInTranslationUnit(Loc, InsertLoc)) &&
              !Sources.isBeforeInTranslationUnit(Loc, MethodLoc))
            InsertLoc = Loc;
        }
      }
    }

    if (InsertLoc.isInvalid() && Method->doesThisDeclarationHaveABody() &&
        Method->getBody() && !Method->isDefaulted()) {
      // For methods with inline definition, add the override keyword at the
      // end of the declaration of the function, but prefer to put it on the
      // same line as the declaration if the beginning brace for the start of
      // the body falls on the next line.
      ReplacementText = " override";
      auto LastTokenIter = std::prev(Tokens.end());
      // When try statement is used instead of compound statement as
      // method body - insert override keyword before it.
      if (LastTokenIter->is(tok::kw_try))
        LastTokenIter = std::prev(LastTokenIter);
      InsertLoc = LastTokenIter->getEndLoc();
    }

    if (!InsertLoc.isValid()) {
      // For declarations marked with "= 0" or "= [default|delete]", the end
      // location will point until after those markings. Therefore, the override
      // keyword shouldn't be inserted at the end, but before the '='.
      if (Tokens.size() > 2 && (GetText(Tokens.back(), Sources) == "0" ||
                                Tokens.back().is(tok::kw_default) ||
                                Tokens.back().is(tok::kw_delete)) &&
          GetText(Tokens[Tokens.size() - 2], Sources) == "=") {
        InsertLoc = Tokens[Tokens.size() - 2].getLocation();
        // Check if we need to insert a space.
        if ((Tokens[Tokens.size() - 2].getFlags() & Token::LeadingSpace) == 0)
          ReplacementText = " override ";
      } else if (GetText(Tokens.back(), Sources) == "ABSTRACT") {
        InsertLoc = Tokens.back().getLocation();
      }
    }

    if (!InsertLoc.isValid()) {
      InsertLoc = FileRange.getEnd();
      ReplacementText = " override";
    }
    Diag << FixItHint::CreateInsertion(InsertLoc, ReplacementText);
  }

  if (HasFinal && HasOverride) {
    SourceLocation OverrideLoc = Method->getAttr<OverrideAttr>()->getLocation();
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getTokenRange(OverrideLoc, OverrideLoc));
  }

  if (HasVirtual) {
    for (Token Tok : Tokens) {
      if (Tok.is(tok::kw_virtual)) {
        Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
            Tok.getLocation(), Tok.getLocation()));
        break;
      }
    }
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
