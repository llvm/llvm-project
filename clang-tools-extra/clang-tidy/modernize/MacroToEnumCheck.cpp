//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroToEnumCheck.h"
#include "../utils/LexerUtils.h"
#include "IntegralLiteralExpressionMatcher.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <cctype>
#include <string>

namespace clang::tidy::modernize {

using utils::lexer::getTokenName;

static bool hasOnlyComments(SourceLocation Loc, const LangOptions &Options,
                            StringRef Text) {
  // Use a lexer to look for tokens; if we find something other than a single
  // hash, then there were intervening tokens between macro definitions.
  const std::string Buffer{Text};
  Lexer Lex(Loc, Options, Buffer.c_str(), Buffer.c_str(),
            Buffer.c_str() + Buffer.size());
  Token Tok;
  bool SeenHash = false;
  while (!Lex.LexFromRawLexer(Tok)) {
    if (Tok.getKind() == tok::hash && !SeenHash) {
      SeenHash = true;
      continue;
    }
    return false;
  }

  // Everything in between was whitespace, so now just look for two blank lines,
  // consisting of two consecutive EOL sequences, either '\n', '\r' or '\r\n'.
  enum class WhiteSpace {
    Nothing,
    CR,
    LF,
    CRLF,
    CRLFCR,
  };

  WhiteSpace State = WhiteSpace::Nothing;
  for (const char C : Text) {
    switch (C) {
    case '\r':
      if (State == WhiteSpace::CR)
        return false;

      State = State == WhiteSpace::CRLF ? WhiteSpace::CRLFCR : WhiteSpace::CR;
      break;

    case '\n':
      if (State == WhiteSpace::LF || State == WhiteSpace::CRLFCR)
        return false;

      State = State == WhiteSpace::CR ? WhiteSpace::CRLF : WhiteSpace::LF;
      break;

    default:
      State = WhiteSpace::Nothing;
      break;
    }
  }

  return true;
}

namespace {

struct EnumMacro {
  EnumMacro(Token Name, const MacroDirective *Directive)
      : Name(Name), Directive(Directive) {}

  Token Name;
  const MacroDirective *Directive;
};

struct MacroExpansion {
  const MacroInfo *Info;
  SourceLocation End;
  bool Complete = true;
  bool Consumed = false;
};

struct MacroState {
  unsigned UnmatchedExpansions = 0;
  const NamespaceDecl *DefinitionNamespace = nullptr;
  // The innermost namespace enclosing every matched use.
  const NamespaceDecl *CommonUseNamespace = nullptr;
  bool HasUses = false;
};

using MacroList = SmallVector<EnumMacro>;

enum class IncludeGuard { None, FileChanged, IfGuard, DefineGuard };

struct FileState {
  FileState() = default;

  int ConditionScopes = 0;
  unsigned int LastLine = 0;
  IncludeGuard GuardScanner = IncludeGuard::None;
  SourceLocation LastMacroLocation;
};

} // namespace

class MacroToEnumCallbacks : public PPCallbacks {
public:
  MacroToEnumCallbacks(MacroToEnumCheck *Check, const LangOptions &LangOptions,
                       const SourceManager &SM)
      : Check(Check), LangOpts(LangOptions), SM(SM) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override {
    clearCurrentEnum(HashLoc);
  }

  // Keep track of macro definitions that look like enums.
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;

  // Undefining an enum-like macro results in the enum set being dropped.
  void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                      const MacroDirective *Undef) override;

  // Conditional compilation clears any adjacent enum-like macros.
  // Macros used in conditional expressions clear any adjacent enum-like
  // macros.
  // Include guards are either
  //   #if !defined(GUARD)
  // or
  //   #ifndef GUARD
  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override {
    conditionStart(Loc);
    checkCondition(ConditionRange);
  }
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    conditionStart(Loc);
    checkName(MacroNameTok);
  }
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    conditionStart(Loc);
    checkName(MacroNameTok);
  }
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override {
    checkCondition(ConditionRange);
  }
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override {
    checkName(MacroNameTok);
  }
  void Elifdef(SourceLocation Loc, SourceRange ConditionRange,
               SourceLocation IfLoc) override {
    PPCallbacks::Elifdef(Loc, ConditionRange, IfLoc);
  }
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override {
    checkName(MacroNameTok);
  }
  void Elifndef(SourceLocation Loc, SourceRange ConditionRange,
                SourceLocation IfLoc) override {
    PPCallbacks::Elifndef(Loc, ConditionRange, IfLoc);
  }
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override;
  void PragmaDirective(SourceLocation Loc,
                       PragmaIntroducerKind Introducer) override;

  void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override;

  // After we've seen everything, issue warnings and fix-its.
  void EndOfMainFile() override;

  void invalidateRange(SourceRange Range);

  void validateExpansionRange(SourceRange Range,
                              const NamespaceDecl *UseNamespace);

  void recordNamespace(const NamespaceDecl *Namespace);

private:
  void newEnum() {
    if (Enums.empty() || !Enums.back().empty())
      Enums.emplace_back();
  }
  bool insideConditional() const {
    return (CurrentFile->GuardScanner == IncludeGuard::DefineGuard &&
            CurrentFile->ConditionScopes > 1) ||
           (CurrentFile->GuardScanner != IncludeGuard::DefineGuard &&
            CurrentFile->ConditionScopes > 0);
  }
  bool isConsecutiveMacro(const MacroDirective *MD) const;
  void rememberLastMacroLocation(const MacroDirective *MD) {
    CurrentFile->LastLine = SM.getSpellingLineNumber(MD->getLocation());
    CurrentFile->LastMacroLocation = Lexer::getLocForEndOfToken(
        MD->getMacroInfo()->getDefinitionEndLoc(), 0, SM, LangOpts);
  }
  void clearLastMacroLocation() {
    CurrentFile->LastLine = 0;
    CurrentFile->LastMacroLocation = SourceLocation{};
  }
  void clearCurrentEnum(SourceLocation Loc);
  void conditionStart(const SourceLocation &Loc);
  void checkCondition(SourceRange ConditionRange);
  void checkName(const Token &MacroNameTok);
  void rememberExpressionName(const Token &Tok);
  void rememberExpressionTokens(ArrayRef<Token> MacroTokens);
  void invalidateExpressionNames();
  void collectArgumentExpansions();
  void validateExpansion(MacroExpansion &Expansion,
                         const NamespaceDecl *UseNamespace);
  void issueDiagnostics();
  void warnMacroEnum(const EnumMacro &Macro) const;
  void fixEnumMacro(const MacroList &MacroList) const;
  bool isInitializer(ArrayRef<Token> MacroTokens);

  MacroToEnumCheck *Check;
  const LangOptions &LangOpts;
  const SourceManager &SM;
  SmallVector<MacroList> Enums;
  llvm::DenseMap<const MacroInfo *, MacroState> MacroStates;
  llvm::DenseMap<SourceLocation, MacroExpansion> Expansions;
  llvm::DenseMap<unsigned, MacroExpansion> ArgumentExpansions;
  bool ArgumentExpansionsCollected = false;
  SmallVector<const MacroInfo *> CandidateDefinitions;
  bool DefinitionsSorted = false;
  SmallVector<FileState> Files;
  std::vector<std::string> ExpressionNames;
  FileState *CurrentFile = nullptr;
};

bool MacroToEnumCallbacks::isConsecutiveMacro(const MacroDirective *MD) const {
  if (CurrentFile->LastMacroLocation.isInvalid())
    return false;

  const SourceLocation Loc = MD->getLocation();
  if (CurrentFile->LastLine + 1 == SM.getSpellingLineNumber(Loc))
    return true;

  const SourceLocation Define =
      SM.translateLineCol(SM.getFileID(Loc), SM.getSpellingLineNumber(Loc), 1);
  const CharSourceRange BetweenMacros{
      SourceRange{CurrentFile->LastMacroLocation, Define}, true};
  const CharSourceRange CharRange =
      Lexer::makeFileCharRange(BetweenMacros, SM, LangOpts);
  const StringRef BetweenText = Lexer::getSourceText(CharRange, SM, LangOpts);
  return hasOnlyComments(Define, LangOpts, BetweenText);
}

void MacroToEnumCallbacks::clearCurrentEnum(SourceLocation Loc) {
  // Only drop the most recent Enum set if the directive immediately follows.
  if (!Enums.empty() && !Enums.back().empty() &&
      SM.getSpellingLineNumber(Loc) == CurrentFile->LastLine + 1)
    Enums.pop_back();

  clearLastMacroLocation();
}

void MacroToEnumCallbacks::conditionStart(const SourceLocation &Loc) {
  ++CurrentFile->ConditionScopes;
  clearCurrentEnum(Loc);
  if (CurrentFile->GuardScanner == IncludeGuard::FileChanged)
    CurrentFile->GuardScanner = IncludeGuard::IfGuard;
}

void MacroToEnumCallbacks::checkCondition(SourceRange Range) {
  for (const Token &Tok : utils::lexer::getRawTokens(
           CharSourceRange::getTokenRange(Range), SM, LangOpts)) {
    if (Tok.is(tok::raw_identifier) && getTokenName(Tok) != "defined")
      checkName(Tok);
  }
}

void MacroToEnumCallbacks::checkName(const Token &MacroNameTok) {
  rememberExpressionName(MacroNameTok);

  StringRef Id = getTokenName(MacroNameTok);
  llvm::erase_if(Enums, [&Id](const MacroList &MacroList) {
    return llvm::any_of(MacroList, [&Id](const EnumMacro &Macro) {
      return getTokenName(Macro.Name) == Id;
    });
  });
}

void MacroToEnumCallbacks::rememberExpressionName(const Token &Tok) {
  const std::string Id = getTokenName(Tok).str();
  const auto Pos = llvm::lower_bound(ExpressionNames, Id);
  if (Pos == ExpressionNames.end() || *Pos != Id)
    ExpressionNames.insert(Pos, Id);
}

void MacroToEnumCallbacks::rememberExpressionTokens(
    ArrayRef<Token> MacroTokens) {
  for (const Token Tok : MacroTokens)
    if (Tok.isAnyIdentifier())
      rememberExpressionName(Tok);
}

void MacroToEnumCallbacks::FileChanged(SourceLocation Loc,
                                       FileChangeReason Reason,
                                       SrcMgr::CharacteristicKind FileType,
                                       FileID PrevFID) {
  newEnum();
  if (Reason == EnterFile) {
    Files.emplace_back();
    if (!SM.isInMainFile(Loc))
      Files.back().GuardScanner = IncludeGuard::FileChanged;
  } else if (Reason == ExitFile) {
    assert(CurrentFile->ConditionScopes == 0);
    Files.pop_back();
  }
  CurrentFile = &Files.back();
}

bool MacroToEnumCallbacks::isInitializer(ArrayRef<Token> MacroTokens) {
  IntegralLiteralExpressionMatcher Matcher(MacroTokens, LangOpts.C99 == 0);
  const bool Matched = Matcher.match();
  const bool IsC = !LangOpts.CPlusPlus;
  if (IsC && (Matcher.largestLiteralSize() != LiteralSize::Int &&
              Matcher.largestLiteralSize() != LiteralSize::UnsignedInt))
    return false;

  return Matched;
}

// Any defined but rejected macro is scanned for identifiers that
// are to be excluded as enums.
void MacroToEnumCallbacks::MacroDefined(const Token &MacroNameTok,
                                        const MacroDirective *MD) {
  // Include guards are never candidates for becoming an enum.
  if (CurrentFile->GuardScanner == IncludeGuard::IfGuard) {
    CurrentFile->GuardScanner = IncludeGuard::DefineGuard;
    return;
  }

  // An earlier definition would still expand the generated enumerator name.
  if (MD->getPrevious())
    checkName(MacroNameTok);

  if (insideConditional())
    return;

  if (SM.getFilename(MD->getLocation()).empty())
    return;

  const MacroInfo *Info = MD->getMacroInfo();
  const ArrayRef<Token> MacroTokens = Info->tokens();
  if (Info->isBuiltinMacro() || MacroTokens.empty())
    return;
  if (Info->isFunctionLike()) {
    rememberExpressionTokens(MacroTokens);
    return;
  }

  if (!isInitializer(MacroTokens))
    return;

  if (!isConsecutiveMacro(MD))
    newEnum();
  Enums.back().emplace_back(MacroNameTok, MD);
  MacroStates.try_emplace(Info);
  CandidateDefinitions.push_back(Info);
  DefinitionsSorted = false;
  rememberLastMacroLocation(MD);
}

// Any macro that is undefined removes all adjacent macros from consideration as
// an enum and starts a new enum scan.
void MacroToEnumCallbacks::MacroUndefined(const Token &MacroNameTok,
                                          const MacroDefinition &MD,
                                          const MacroDirective *Undef) {
  rememberExpressionName(MacroNameTok);

  const auto MatchesToken = [&MacroNameTok](const EnumMacro &Macro) {
    return getTokenName(Macro.Name) == getTokenName(MacroNameTok);
  };

  const auto *It =
      llvm::find_if(Enums, [MatchesToken](const MacroList &MacroList) {
        return llvm::any_of(MacroList, MatchesToken);
      });
  if (It != Enums.end())
    Enums.erase(It);

  clearLastMacroLocation();
  CurrentFile->GuardScanner = IncludeGuard::None;
}

void MacroToEnumCallbacks::Endif(SourceLocation Loc, SourceLocation IfLoc) {
  // The if directive for the include guard isn't counted in the
  // ConditionScopes.
  if (CurrentFile->ConditionScopes == 0 &&
      CurrentFile->GuardScanner == IncludeGuard::DefineGuard)
    return;

  // We don't need to clear the current enum because the start of the
  // conditional block already took care of that.
  assert(CurrentFile->ConditionScopes > 0);
  --CurrentFile->ConditionScopes;
}

template <size_t N>
static bool textEquals(const char (&Needle)[N], const char *HayStack) {
  return StringRef{HayStack, N - 1} == Needle;
}

template <size_t N> static size_t len(const char (&)[N]) { return N - 1; }

void MacroToEnumCallbacks::PragmaDirective(SourceLocation Loc,
                                           PragmaIntroducerKind Introducer) {
  if (CurrentFile->GuardScanner != IncludeGuard::FileChanged)
    return;

  bool Invalid = false;
  const char *Text = SM.getCharacterData(
      Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts), &Invalid);
  if (Invalid)
    return;

  while (*Text && std::isspace(*Text))
    ++Text;

  if (textEquals("pragma", Text))
    return;

  Text += len("pragma");
  while (*Text && std::isspace(*Text))
    ++Text;

  if (textEquals("once", Text))
    CurrentFile->GuardScanner = IncludeGuard::IfGuard;
}

void MacroToEnumCallbacks::MacroExpands(const Token &MacroNameTok,
                                        const MacroDefinition &MD,
                                        SourceRange Range,
                                        const MacroArgs *Args) {
  // Ignore function-like macro expansions.
  if (Args != nullptr)
    return;

  const MacroInfo *Info = MD.getMacroInfo();
  const auto Candidate = MacroStates.find(Info);
  if (Candidate == MacroStates.end())
    return;

  if (Expansions
          .try_emplace(Range.getBegin(), MacroExpansion{Info, Range.getEnd()})
          .second)
    ++Candidate->second.UnmatchedExpansions;
}

void MacroToEnumCallbacks::invalidateExpressionNames() {
  for (const std::string &Id : ExpressionNames) {
    llvm::erase_if(Enums, [Id](const MacroList &MacroList) {
      return llvm::any_of(MacroList, [&Id](const EnumMacro &Macro) {
        return getTokenName(Macro.Name) == Id;
      });
    });
  }
}

static const NamespaceDecl *getNamespaceContext(const DeclContext *Context) {
  for (; Context; Context = Context->getParent())
    if (const auto *Namespace = dyn_cast<NamespaceDecl>(Context))
      return Namespace->getCanonicalDecl();
  return nullptr;
}

static const NamespaceDecl *getEnclosingNamespace(const Decl *Declaration) {
  return getNamespaceContext(Declaration->getDeclContext());
}

static const NamespaceDecl *getUseNamespace(const Decl *Use,
                                            SourceLocation Location,
                                            const SourceManager &SM) {
  for (const Decl *Declaration = Use; Declaration;) {
    const DeclContext *Context = Declaration->getDeclContext();
    const DeclContext *Lexical = Declaration->getLexicalDeclContext();
    if (Context != Lexical) {
      // A qualified declaration's leading type is still in its lexical scope.
      bool InSemanticScope = false;
      if (const auto *Function = dyn_cast<FunctionDecl>(Declaration)) {
        const SourceLocation NameEnd = Function->getNameInfo().getEndLoc();
        SourceLocation ScopeBegin = NameEnd;
        if (const auto *Arguments =
                Function->getTemplateSpecializationArgsAsWritten())
          ScopeBegin = Arguments->getRAngleLoc();
        InSemanticScope = NameEnd.isValid() && ScopeBegin.isValid() &&
                          !SM.isBeforeInTranslationUnit(ScopeBegin, NameEnd) &&
                          !SM.isBeforeInTranslationUnit(Location, ScopeBegin);
      } else {
        SourceRange ScopeRange;
        if (const auto *Record = dyn_cast<RecordDecl>(Declaration)) {
          ScopeRange = Record->getBraceRange();
        } else if (const auto *Variable = dyn_cast<VarDecl>(Declaration)) {
          if (const Expr *Initializer = Variable->getInit())
            ScopeRange = Initializer->getSourceRange();
        }
        InSemanticScope = ScopeRange.isValid() &&
                          SM.isPointWithin(Location, ScopeRange.getBegin(),
                                           ScopeRange.getEnd());
      }
      if (!InSemanticScope)
        return getNamespaceContext(Lexical);
    }
    Declaration = Context ? Decl::castFromDeclContext(Context) : nullptr;
  }
  return getEnclosingNamespace(Use);
}

static bool isWithinNamespace(const NamespaceDecl *UseNamespace,
                              const NamespaceDecl *DefinitionNamespace) {
  if (!DefinitionNamespace)
    return true;
  for (const NamespaceDecl *Namespace = UseNamespace; Namespace;
       Namespace = getEnclosingNamespace(Namespace)) {
    if (Namespace == DefinitionNamespace)
      return true;
  }
  return false;
}

static const NamespaceDecl *getCommonNamespace(const NamespaceDecl *Left,
                                               const NamespaceDecl *Right) {
  if (!Left || !Right)
    return nullptr;
  for (const NamespaceDecl *Namespace = Left; Namespace;
       Namespace = getEnclosingNamespace(Namespace)) {
    if (isWithinNamespace(Right, Namespace))
      return Namespace;
  }
  return nullptr;
}

void MacroToEnumCallbacks::recordNamespace(const NamespaceDecl *Namespace) {
  if (Namespace->getBeginLoc().isInvalid() ||
      Namespace->getEndLoc().isInvalid())
    return;

  if (!DefinitionsSorted) {
    llvm::sort(CandidateDefinitions,
               [this](const MacroInfo *Left, const MacroInfo *Right) {
                 return SM.isBeforeInTranslationUnit(Left->getDefinitionLoc(),
                                                     Right->getDefinitionLoc());
               });
    DefinitionsSorted = true;
  }

  const SourceRange Range =
      SM.getExpansionRange(Namespace->getSourceRange()).getAsRange();
  const auto *It = llvm::lower_bound(
      CandidateDefinitions, Range.getBegin(),
      [this](const MacroInfo *Info, SourceLocation Location) {
        return SM.isBeforeInTranslationUnit(Info->getDefinitionLoc(), Location);
      });
  const NamespaceDecl *Canonical = Namespace->getCanonicalDecl();
  for (; It != CandidateDefinitions.end() &&
         SM.isPointWithin((*It)->getDefinitionLoc(), Range.getBegin(),
                          Range.getEnd());
       ++It) {
    MacroState &State = MacroStates.find(*It)->second;
    // Nested namespaces can share source ranges, as in namespace A::B {}.
    if (isWithinNamespace(Canonical, State.DefinitionNamespace))
      State.DefinitionNamespace = Canonical;
  }
}

void MacroToEnumCallbacks::EndOfMainFile() {
  invalidateExpressionNames();
  llvm::erase_if(Enums, [this](const MacroList &Macros) {
    return llvm::any_of(Macros, [this](const EnumMacro &Macro) {
      const MacroState &State =
          MacroStates.find(Macro.Directive->getMacroInfo())->second;
      return State.UnmatchedExpansions != 0 ||
             (State.HasUses && !isWithinNamespace(State.CommonUseNamespace,
                                                  State.DefinitionNamespace));
    });
  });
  issueDiagnostics();
}

void MacroToEnumCallbacks::invalidateRange(SourceRange Range) {
  // A declaration can end in a macro expansion. Compare file locations so it
  // does not also invalidate unrelated macros defined after the declaration.
  Range = SM.getExpansionRange(Range).getAsRange();
  llvm::erase_if(Enums, [this, Range](const MacroList &MacroList) {
    return llvm::any_of(MacroList, [this, Range](const EnumMacro &Macro) {
      return SM.isPointWithin(Macro.Directive->getLocation(), Range.getBegin(),
                              Range.getEnd());
    });
  });
}

void MacroToEnumCallbacks::collectArgumentExpansions() {
  if (ArgumentExpansionsCollected)
    return;
  ArgumentExpansionsCollected = true;

  // Argument prescanning expands a candidate once, but each substituted copy
  // must be validated independently. Source entries are created parent first.
  for (unsigned I = 1, Size = SM.local_sloc_entry_size(); I < Size; ++I) {
    const auto &Entry = SM.getLocalSLocEntry(I);
    if (!Entry.isExpansion() || !Entry.getExpansion().isMacroArgExpansion())
      continue;

    const SourceLocation Spelling = Entry.getExpansion().getSpellingLoc();
    if (!Spelling.isMacroID())
      continue;

    const unsigned NextOffset = I + 1 == Size
                                    ? SM.getNextLocalOffset()
                                    : SM.getLocalSLocEntry(I + 1).getOffset();
    const unsigned Length = NextOffset - Entry.getOffset() - 1;
    const SourceLocation SpellingEnd = Spelling.getLocWithOffset(Length);
    const FileID SpellingFile = SM.getFileID(Spelling);
    MacroExpansion *Parent = nullptr;
    bool Complete = false;
    if (SM.isMacroArgExpansion(Spelling)) {
      const auto Argument =
          ArgumentExpansions.find(SM.getSLocEntry(SpellingFile).getOffset());
      if (Argument == ArgumentExpansions.end())
        continue;
      Parent = &Argument->second;
      Complete = Parent->Complete && SM.getFileOffset(Spelling) == 0 &&
                 Length == SM.getFileIDSize(SpellingFile);
    } else {
      const SourceRange Range =
          SM.getImmediateExpansionRange(Spelling).getAsRange();
      const auto Expansion = Expansions.find(Range.getBegin());
      if (Expansion == Expansions.end() ||
          Expansion->second.End != Range.getEnd())
        continue;
      Parent = &Expansion->second;
      Complete = SM.isAtStartOfImmediateMacroExpansion(Spelling) &&
                 SM.isInFileID(SpellingEnd, SpellingFile) &&
                 SM.isAtEndOfImmediateMacroExpansion(SpellingEnd);
    }

    const MacroInfo *Info = Parent->Info;
    MacroState &State = MacroStates.find(Info)->second;
    // Forwarding replaces the parent's obligation with one for each child.
    if (!Parent->Consumed) {
      assert(State.UnmatchedExpansions != 0);
      --State.UnmatchedExpansions;
      Parent->Consumed = true;
    }
    ArgumentExpansions.try_emplace(Entry.getOffset(),
                                   MacroExpansion{Info, SpellingEnd, Complete});
    ++State.UnmatchedExpansions;
  }
}

void MacroToEnumCallbacks::validateExpansion(
    MacroExpansion &Expansion, const NamespaceDecl *UseNamespace) {
  if (Expansion.Consumed || !Expansion.Complete)
    return;

  MacroState &State = MacroStates.find(Expansion.Info)->second;
  assert(State.UnmatchedExpansions != 0);
  --State.UnmatchedExpansions;
  State.CommonUseNamespace =
      State.HasUses ? getCommonNamespace(State.CommonUseNamespace, UseNamespace)
                    : UseNamespace;
  State.HasUses = true;
  // A template instantiation can revisit this expression's source range.
  Expansion.Consumed = true;
}

void MacroToEnumCallbacks::validateExpansionRange(
    SourceRange Range, const NamespaceDecl *UseNamespace) {
  collectArgumentExpansions();
  SourceLocation Begin = Range.getBegin();
  SourceLocation End = Range.getEnd();
  while (Begin.isMacroID() && End.isMacroID() &&
         SM.getFileID(Begin) == SM.getFileID(End)) {
    const unsigned Length =
        Lexer::MeasureTokenLength(SM.getSpellingLoc(End), SM, LangOpts);
    const SourceLocation TokenEnd = End.getLocWithOffset(Length);
    if (Length == 0 || !SM.isInFileID(TokenEnd, SM.getFileID(End)))
      return;

    if (SM.isMacroArgExpansion(Begin)) {
      const auto Argument = ArgumentExpansions.find(
          SM.getSLocEntry(SM.getFileID(Begin)).getOffset());
      if (Argument != ArgumentExpansions.end() &&
          SM.getFileOffset(Begin) == 0 &&
          SM.getImmediateSpellingLoc(TokenEnd) == Argument->second.End)
        validateExpansion(Argument->second, UseNamespace);
      return;
    }

    // An inner expression can be complete without consuming its enclosing
    // object-like macro expansion.
    SourceLocation ExpansionBegin, ExpansionEnd;
    if (!SM.isAtStartOfImmediateMacroExpansion(Begin, &ExpansionBegin) ||
        !SM.isAtEndOfImmediateMacroExpansion(TokenEnd, &ExpansionEnd))
      return;

    const auto Expansion = Expansions.find(ExpansionBegin);
    if (Expansion != Expansions.end() && Expansion->second.End == ExpansionEnd)
      validateExpansion(Expansion->second, UseNamespace);
    Begin = ExpansionBegin;
    End = ExpansionEnd;
  }
}

void MacroToEnumCallbacks::issueDiagnostics() {
  for (const MacroList &MacroList : Enums) {
    if (MacroList.empty())
      continue;

    for (const EnumMacro &Macro : MacroList)
      warnMacroEnum(Macro);

    fixEnumMacro(MacroList);
  }
}

void MacroToEnumCallbacks::warnMacroEnum(const EnumMacro &Macro) const {
  Check->diag(Macro.Directive->getLocation(),
              "macro '%0' defines an integral constant; prefer an enum instead")
      << getTokenName(Macro.Name);
}

void MacroToEnumCallbacks::fixEnumMacro(const MacroList &MacroList) const {
  SourceLocation Begin =
      MacroList.front().Directive->getMacroInfo()->getDefinitionLoc();
  const StringRef LineEnding =
      SM.getBufferData(SM.getFileID(Begin)).detectEOL();

  Begin = SM.translateLineCol(SM.getFileID(Begin),
                              SM.getSpellingLineNumber(Begin), 1);
  const DiagnosticBuilder Diagnostic =
      Check->diag(Begin, "replace macro with enum")
      << FixItHint::CreateInsertion(Begin,
                                    (llvm::Twine("enum {") + LineEnding).str());

  for (size_t I = 0U; I < MacroList.size(); ++I) {
    const EnumMacro &Macro = MacroList[I];
    const SourceLocation DefineEnd =
        Macro.Directive->getMacroInfo()->getDefinitionLoc();
    const SourceLocation DefineBegin = SM.translateLineCol(
        SM.getFileID(DefineEnd), SM.getSpellingLineNumber(DefineEnd), 1);
    CharSourceRange DefineRange;
    DefineRange.setBegin(DefineBegin);
    DefineRange.setEnd(DefineEnd);
    Diagnostic << FixItHint::CreateRemoval(DefineRange);

    const SourceLocation NameEnd = Lexer::getLocForEndOfToken(
        Macro.Directive->getMacroInfo()->getDefinitionLoc(), 0, SM, LangOpts);
    Diagnostic << FixItHint::CreateInsertion(NameEnd, " =");

    const SourceLocation ValueEnd = Lexer::getLocForEndOfToken(
        Macro.Directive->getMacroInfo()->getDefinitionEndLoc(), 0, SM,
        LangOpts);
    if (I < MacroList.size() - 1)
      Diagnostic << FixItHint::CreateInsertion(ValueEnd, ",");
  }

  SourceLocation End = Lexer::getLocForEndOfToken(
      MacroList.back().Directive->getMacroInfo()->getDefinitionEndLoc(), 0, SM,
      LangOpts);
  End = SM.translateLineCol(SM.getFileID(End),
                            SM.getSpellingLineNumber(End) + 1, 1);
  Diagnostic << FixItHint::CreateInsertion(
      End, (llvm::Twine("};") + LineEnding).str());
}

void MacroToEnumCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  auto Callback =
      std::make_unique<MacroToEnumCallbacks>(this, getLangOpts(), SM);
  PPCallback = Callback.get();
  PP->addPPCallbacks(std::move(Callback));
}

void MacroToEnumCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  using namespace ast_matchers;
  // Inspect declarations inside namespace and linkage containers. Namespace
  // ownership is checked separately against each expansion's context.
  const auto TopLevelDecl = hasParent(
      decl(anyOf(translationUnitDecl(), linkageSpecDecl(), namespaceDecl())));
  Finder->addMatcher(
      decl(TopLevelDecl, unless(anyOf(linkageSpecDecl(), namespaceDecl())))
          .bind("top"),
      this);
  Finder->addMatcher(namespaceDecl().bind("namespace"), this);
  Finder->addMatcher(expr(hasAncestor(decl().bind("use"))).bind("expr"), this);
}

static bool isValid(SourceRange Range) {
  return Range.getBegin().isValid() && Range.getEnd().isValid();
}

static bool empty(SourceRange Range) {
  return Range.getBegin() == Range.getEnd();
}

void MacroToEnumCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (const auto *Namespace =
          Result.Nodes.getNodeAs<NamespaceDecl>("namespace"))
    PPCallback->recordNamespace(Namespace);

  if (const auto *E = Result.Nodes.getNodeAs<Expr>("expr");
      E && E->getBeginLoc().isMacroID() && E->getEndLoc().isMacroID()) {
    const auto *Use = Result.Nodes.getNodeAs<Decl>("use");
    assert(Use);
    PPCallback->validateExpansionRange(
        E->getSourceRange(),
        getUseNamespace(Use, E->getBeginLoc(), *Result.SourceManager));
  }

  auto *TLDecl = Result.Nodes.getNodeAs<Decl>("top");
  if (TLDecl == nullptr)
    return;

  SourceRange Range = TLDecl->getSourceRange();
  if (auto *TemplateFn = Result.Nodes.getNodeAs<FunctionTemplateDecl>("top");
      TemplateFn && TemplateFn->isThisDeclarationADefinition() &&
      TemplateFn->hasBody())
    Range = SourceRange{TemplateFn->getBeginLoc(),
                        TemplateFn->getUnderlyingDecl()->getBodyRBrace()};

  if (isValid(Range) && !empty(Range))
    PPCallback->invalidateRange(Range);
}

} // namespace clang::tidy::modernize
