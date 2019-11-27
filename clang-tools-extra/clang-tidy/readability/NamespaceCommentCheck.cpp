//===--- NamespaceCommentCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NamespaceCommentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {
class NamespaceCommentPPCallbacks : public PPCallbacks {
public:
  NamespaceCommentPPCallbacks(Preprocessor *PP, NamespaceCommentCheck *Check)
      : PP(PP), Check(Check) {}

  void MacroDefined(const Token &MacroNameTok, const MacroDirective *MD) {
    // Record all defined macros. We store the whole token to compare names
    // later.

    const MacroInfo * MI = MD->getMacroInfo();

    if (MI->isFunctionLike())
      return;

    std::string ValueBuffer;
    llvm::raw_string_ostream Value(ValueBuffer);

    SmallString<128> SpellingBuffer;
    bool First = true;
    for (const auto &T : MI->tokens()) {
      if (!First && T.hasLeadingSpace())
        Value << ' ';

      Value << PP->getSpelling(T, SpellingBuffer);
      First = false;
    }

    Check->addMacro(MacroNameTok.getIdentifierInfo()->getName().str(),
                    Value.str());
  }

private:
  Preprocessor *PP;
  NamespaceCommentCheck *Check;
};
} // namespace

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
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (getLangOpts().CPlusPlus)
    Finder->addMatcher(namespaceDecl().bind("namespace"), this);
}

void NamespaceCommentCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<NamespaceCommentPPCallbacks>(PP, this));
}

static bool locationsInSameFile(const SourceManager &Sources,
                                SourceLocation Loc1, SourceLocation Loc2) {
  return Loc1.isFileID() && Loc2.isFileID() &&
         Sources.getFileID(Loc1) == Sources.getFileID(Loc2);
}

std::string NamespaceCommentCheck::getNamespaceComment(const NamespaceDecl *ND,
                                                       bool InsertLineBreak) {
  std::string Fix = "// namespace";
  if (!ND->isAnonymousNamespace()) {
    bool IsNamespaceMacroExpansion;
    StringRef MacroDefinition;
    std::tie(IsNamespaceMacroExpansion, MacroDefinition) =
        isNamespaceMacroExpansion(ND->getName());

    Fix.append(" ").append(IsNamespaceMacroExpansion ? MacroDefinition
                                                     : ND->getName());
  }
  if (InsertLineBreak)
    Fix.append("\n");
  return Fix;
}

std::string
NamespaceCommentCheck::getNamespaceComment(const std::string &NameSpaceName,
                                           bool InsertLineBreak) {
  std::string Fix = "// namespace ";
  Fix.append(NameSpaceName);
  if (InsertLineBreak)
    Fix.append("\n");
  return Fix;
}

void NamespaceCommentCheck::addMacro(const std::string &Name,
                                     const std::string &Value) noexcept {
  Macros.emplace_back(Name, Value);
}

bool NamespaceCommentCheck::isNamespaceMacroDefinition(
    const StringRef NameSpaceName) {
  return llvm::any_of(Macros, [&NameSpaceName](const auto &Macro) {
    return NameSpaceName == Macro.first;
  });
}

std::tuple<bool, StringRef> NamespaceCommentCheck::isNamespaceMacroExpansion(
    const StringRef NameSpaceName) {
  const auto &MacroIt =
      llvm::find_if(Macros, [&NameSpaceName](const auto &Macro) {
        return NameSpaceName == Macro.second;
      });

  const bool IsNamespaceMacroExpansion = Macros.end() != MacroIt;

  return std::make_tuple(IsNamespaceMacroExpansion,
                         IsNamespaceMacroExpansion ? StringRef(MacroIt->first)
                                                   : NameSpaceName);
}

void NamespaceCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ND = Result.Nodes.getNodeAs<NamespaceDecl>("namespace");
  const SourceManager &Sources = *Result.SourceManager;

  if (!locationsInSameFile(Sources, ND->getBeginLoc(), ND->getRBraceLoc()))
    return;

  // Don't require closing comments for namespaces spanning less than certain
  // number of lines.
  unsigned StartLine = Sources.getSpellingLineNumber(ND->getBeginLoc());
  unsigned EndLine = Sources.getSpellingLineNumber(ND->getRBraceLoc());
  if (EndLine - StartLine + 1 <= ShortNamespaceLines)
    return;

  // Find next token after the namespace closing brace.
  SourceLocation AfterRBrace = ND->getRBraceLoc().getLocWithOffset(1);
  SourceLocation Loc = AfterRBrace;
  Token Tok;
  SourceLocation LBracketLocation = ND->getLocation();
  SourceLocation NestedNamespaceBegin = LBracketLocation;

  // Currently for nested namepsace (n1::n2::...) the AST matcher will match foo
  // then bar instead of a single match. So if we got a nested namespace we have
  // to skip the next ones.
  for (const auto &EndOfNameLocation : Ends) {
    if (Sources.isBeforeInTranslationUnit(NestedNamespaceBegin,
                                          EndOfNameLocation))
      return;
  }

  // Ignore macros
  if (!ND->getLocation().isMacroID()) {
    while (Lexer::getRawToken(LBracketLocation, Tok, Sources, getLangOpts()) ||
           !Tok.is(tok::l_brace)) {
      LBracketLocation = LBracketLocation.getLocWithOffset(1);
    }
  }

  // FIXME: This probably breaks on comments between the namespace and its '{'.
  auto TextRange =
      Lexer::getAsCharRange(SourceRange(NestedNamespaceBegin, LBracketLocation),
                            Sources, getLangOpts());
  StringRef NestedNamespaceName =
      Lexer::getSourceText(TextRange, Sources, getLangOpts())
          .rtrim('{') // Drop the { itself.
          .rtrim();   // Drop any whitespace before it.
  bool IsNested = NestedNamespaceName.contains(':');

  if (IsNested)
    Ends.push_back(LBracketLocation);
  else
    NestedNamespaceName = ND->getName();

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

      // Don't allow to use macro expansion in closing comment.
      // FIXME: Use Structured Bindings once C++17 features will be enabled.
      bool IsNamespaceMacroExpansion;
      StringRef MacroDefinition;
      std::tie(IsNamespaceMacroExpansion, MacroDefinition) =
          isNamespaceMacroExpansion(NamespaceNameInComment);

      if (IsNested && NestedNamespaceName == NamespaceNameInComment) {
        // C++17 nested namespace.
        return;
      } else if ((ND->isAnonymousNamespace() &&
                  NamespaceNameInComment.empty()) ||
                 (((ND->getNameAsString() == NamespaceNameInComment) &&
                   Anonymous.empty()) &&
                  !IsNamespaceMacroExpansion)) {
        // Check if the namespace in the comment is the same.
        // FIXME: Maybe we need a strict mode, where we always fix namespace
        // comments with different format.
        return;
      }

      // Allow using macro definitions in closing comment.
      if (isNamespaceMacroDefinition(NamespaceNameInComment))
        return;

      // Otherwise we need to fix the comment.
      NeedLineBreak = Comment.startswith("/*");
      OldCommentRange =
          SourceRange(AfterRBrace, Loc.getLocWithOffset(Tok.getLength()));

      if (IsNamespaceMacroExpansion) {
        Message = (llvm::Twine("%0 ends with a comment that refers to an "
                               "expansion of macro"))
                      .str();
        NestedNamespaceName = MacroDefinition;
      } else {
        Message = (llvm::Twine("%0 ends with a comment that refers to a "
                               "wrong namespace '") +
                   NamespaceNameInComment + "'")
                      .str();
      }

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

  // Print Macro definition instead of expansion.
  // FIXME: Use Structured Bindings once C++17 features will be enabled.
  bool IsNamespaceMacroExpansion;
  StringRef MacroDefinition;
  std::tie(IsNamespaceMacroExpansion, MacroDefinition) =
      isNamespaceMacroExpansion(NestedNamespaceName);

  if (IsNamespaceMacroExpansion)
    NestedNamespaceName = MacroDefinition;

  std::string NamespaceName =
      ND->isAnonymousNamespace()
          ? "anonymous namespace"
          : ("namespace '" + NestedNamespaceName.str() + "'");

  // Place diagnostic at an old comment, or closing brace if we did not have it.
  SourceLocation DiagLoc =
      OldCommentRange.getBegin() != OldCommentRange.getEnd()
          ? OldCommentRange.getBegin()
          : ND->getRBraceLoc();

  diag(DiagLoc, Message)
      << NamespaceName
      << FixItHint::CreateReplacement(
             CharSourceRange::getCharRange(OldCommentRange),
             std::string(SpacesBeforeComments, ' ') +
                 (IsNested
                      ? getNamespaceComment(NestedNamespaceName, NeedLineBreak)
                      : getNamespaceComment(ND, NeedLineBreak)));
  diag(ND->getLocation(), "%0 starts here", DiagnosticIDs::Note)
      << NamespaceName;
}

} // namespace readability
} // namespace tidy
} // namespace clang
