//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroConditionCheck.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace clang::tidy::bugprone {

namespace {
class MacroConditionCallbacks : public PPCallbacks {
public:
  MacroConditionCallbacks(MacroConditionCheck *Check, const SourceManager &SM,
                          Preprocessor &PP)
      : Check(Check), SM(SM), PP(PP) {}

  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override;
  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override;
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override;
  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override;
  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override;
  void Elifdef(SourceLocation Loc, SourceRange ConditionRange,
               SourceLocation IfLoc) override;
  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override;
  void Elifndef(SourceLocation Loc, SourceRange ConditionRange,
                SourceLocation IfLoc) override;
  void Else(SourceLocation Loc, SourceLocation IfLoc) override;
  void Endif(SourceLocation Loc, SourceLocation IfLoc) override;
  void SourceRangeSkipped(SourceRange Range, SourceLocation EndifLoc) override;

private:
  enum class ReferenceKind { Definition, NegatedDefinition, Value };

  struct MacroReference {
    std::string Name;
    SourceLocation Loc;
    ReferenceKind Kind;
  };

  struct MacroUsage {
    SourceLocation DefinitionTestLoc;
    SourceLocation ValueTestLoc;
    bool Diagnosed = false;
  };

  struct DefinitionGuard {
    std::string Name;
    FileID File;
  };

  using ConditionReferences = SmallVector<MacroReference, 4>;
  using DefinitionGuards = SmallVector<DefinitionGuard, 2>;

  SmallVector<Token, 16> tokensInCondition(SourceRange ConditionRange) const;
  ConditionReferences referencesInCondition(SourceRange ConditionRange) const;
  DefinitionGuards
  definitionGuardsInCondition(SourceRange ConditionRange) const;
  std::optional<std::string>
  defaultedMacroInSkippedRange(SourceRange Range) const;
  MacroReference referenceFromRange(SourceRange Range,
                                    SourceLocation Loc) const;
  bool branchUnconditionallyErrors(SourceLocation Loc) const;
  bool isEnclosedByDefinitionGuard(const MacroReference &Reference) const;
  void beginConditional(DefinitionGuards Guards);
  void changeConditional(DefinitionGuards Guards);
  bool isIgnoredIdentifier(StringRef Name) const;
  void processReferences(const ConditionReferences &References);
  void processDefinitionReference(StringRef Name, SourceLocation Loc);
  void checkReference(const MacroReference &Reference);

  using FileUsages = llvm::DenseMap<FileID, MacroUsage>;
  llvm::DenseMap<const MacroInfo *, FileUsages> MacroUsages;
  SmallVector<DefinitionGuards, 8> ConditionalGuards;
  MacroConditionCheck *Check;
  const SourceManager &SM;
  Preprocessor &PP;
};

} // namespace

static StringRef getTokenName(const Token &Tok) {
  if (Tok.is(tok::raw_identifier))
    return Tok.getRawIdentifier();
  if (const IdentifierInfo *Info = Tok.getIdentifierInfo())
    return Info->getName();
  return {};
}

static bool skipFunctionLikeInvocation(ArrayRef<Token> Tokens, size_t &Index) {
  if (Index + 1 >= Tokens.size() || Tokens[Index + 1].isNot(tok::l_paren))
    return false;

  unsigned ParenthesisDepth = 0;
  do {
    ++Index;
    if (Tokens[Index].is(tok::l_paren))
      ++ParenthesisDepth;
    else if (Tokens[Index].is(tok::r_paren))
      --ParenthesisDepth;
  } while (Index + 1 < Tokens.size() && ParenthesisDepth != 0);
  return true;
}

static bool isNegatedDefined(ArrayRef<Token> Tokens, size_t Index) {
  unsigned Negations = 0;
  while (Index > 0) {
    while (Index > 0 && Tokens[Index - 1].is(tok::l_paren))
      --Index;
    if (Index == 0 || Tokens[Index - 1].isNot(tok::exclaim))
      break;
    --Index;
    ++Negations;
  }
  return Negations % 2 != 0;
}

static bool isStandardPredefinedMacro(StringRef Name) {
  return Name == "__cplusplus" || Name == "__DATE__" || Name == "__FILE__" ||
         Name == "__LINE__" || Name == "__TIME__" ||
         Name.starts_with("__cpp_") || Name.starts_with("__STDC_") ||
         Name.starts_with("__STDCPP_");
}

SmallVector<Token, 16>
MacroConditionCallbacks::tokensInCondition(SourceRange ConditionRange) const {
  const SourceLocation BeginLoc = SM.getExpansionLoc(ConditionRange.getBegin());
  if (BeginLoc.isInvalid())
    return {};

  const std::pair<FileID, unsigned> Decomposed = SM.getDecomposedLoc(BeginLoc);
  bool Invalid = false;
  StringRef Buffer = SM.getBufferData(Decomposed.first, &Invalid);
  if (Invalid || Decomposed.second >= Buffer.size())
    return {};

  size_t End = Decomposed.second;
  while (End < Buffer.size()) {
    if (Buffer[End] != '\r' && Buffer[End] != '\n') {
      ++End;
      continue;
    }

    const size_t Newline = End;
    if (Newline > Decomposed.second && Buffer[Newline - 1] == '\\') {
      if (Buffer[End] == '\r' && End + 1 < Buffer.size() &&
          Buffer[End + 1] == '\n')
        ++End;
      ++End;
      continue;
    }
    break;
  }

  Lexer Lex(SM.getLocForStartOfFile(Decomposed.first), PP.getLangOpts(),
            Buffer.begin(), Buffer.begin() + Decomposed.second, Buffer.end());
  SmallVector<Token, 16> Tokens;
  Token Tok;
  bool AtEnd = false;
  do {
    AtEnd = Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::eof) || SM.getFileOffset(Tok.getLocation()) >= End)
      break;
    Tokens.push_back(Tok);
  } while (!AtEnd);
  return Tokens;
}

MacroConditionCallbacks::ConditionReferences
MacroConditionCallbacks::referencesInCondition(
    SourceRange ConditionRange) const {
  ConditionReferences References;
  const SmallVector<Token, 16> Tokens = tokensInCondition(ConditionRange);

  for (size_t Index = 0; Index < Tokens.size(); ++Index) {
    const Token &Current = Tokens[Index];
    if (!Current.is(tok::raw_identifier))
      continue;

    StringRef Name = Current.getRawIdentifier();
    if (Name != "defined") {
      if (skipFunctionLikeInvocation(Tokens, Index))
        continue;
      if (!isIgnoredIdentifier(Name))
        References.push_back(
            {Name.str(), Current.getLocation(), ReferenceKind::Value});
      continue;
    }

    const SourceLocation DefinedLoc = Current.getLocation();
    const bool IsNegated = isNegatedDefined(Tokens, Index);
    ++Index;
    if (Index < Tokens.size() && Tokens[Index].is(tok::l_paren))
      ++Index;
    if (Index < Tokens.size() && Tokens[Index].is(tok::raw_identifier))
      References.push_back({Tokens[Index].getRawIdentifier().str(), DefinedLoc,
                            IsNegated ? ReferenceKind::NegatedDefinition
                                      : ReferenceKind::Definition});
  }
  return References;
}

static bool
parseDefinitionGuardExpression(ArrayRef<Token> Tokens, size_t &Index,
                               SmallVectorImpl<std::string> &Guards);

static bool parseDefinitionGuardPrimary(ArrayRef<Token> Tokens, size_t &Index,
                                        SmallVectorImpl<std::string> &Guards) {
  if (Index >= Tokens.size())
    return false;

  if (Tokens[Index].is(tok::l_paren)) {
    ++Index;
    if (!parseDefinitionGuardExpression(Tokens, Index, Guards) ||
        Index >= Tokens.size() || Tokens[Index].isNot(tok::r_paren))
      return false;
    ++Index;
    return true;
  }

  if (getTokenName(Tokens[Index]) != "defined")
    return false;
  ++Index;

  const bool Parenthesized =
      Index < Tokens.size() && Tokens[Index].is(tok::l_paren);
  if (Parenthesized)
    ++Index;
  if (Index >= Tokens.size() || Tokens[Index].isNot(tok::raw_identifier))
    return false;
  Guards.push_back(Tokens[Index++].getRawIdentifier().str());
  if (Parenthesized) {
    if (Index >= Tokens.size() || Tokens[Index].isNot(tok::r_paren))
      return false;
    ++Index;
  }
  return true;
}

static bool
parseDefinitionGuardExpression(ArrayRef<Token> Tokens, size_t &Index,
                               SmallVectorImpl<std::string> &Guards) {
  if (!parseDefinitionGuardPrimary(Tokens, Index, Guards))
    return false;
  while (Index < Tokens.size() && Tokens[Index].is(tok::ampamp)) {
    ++Index;
    if (!parseDefinitionGuardPrimary(Tokens, Index, Guards))
      return false;
  }
  return true;
}

MacroConditionCallbacks::DefinitionGuards
MacroConditionCallbacks::definitionGuardsInCondition(
    SourceRange ConditionRange) const {
  const SmallVector<Token, 16> Tokens = tokensInCondition(ConditionRange);
  SmallVector<std::string, 2> Names;
  size_t Index = 0;
  if (!parseDefinitionGuardExpression(Tokens, Index, Names) ||
      Index != Tokens.size())
    return {};

  DefinitionGuards Guards;
  const FileID File =
      SM.getFileID(SM.getSpellingLoc(ConditionRange.getBegin()));
  for (std::string &Name : Names)
    Guards.push_back({std::move(Name), File});
  return Guards;
}

bool MacroConditionCallbacks::branchUnconditionallyErrors(
    SourceLocation Loc) const {
  const SourceLocation BeginLoc = SM.getExpansionLoc(Loc);
  if (BeginLoc.isInvalid())
    return false;

  const std::pair<FileID, unsigned> Decomposed = SM.getDecomposedLoc(BeginLoc);
  bool Invalid = false;
  StringRef Buffer = SM.getBufferData(Decomposed.first, &Invalid);
  if (Invalid || Decomposed.second >= Buffer.size())
    return false;

  size_t Body = Decomposed.second;
  while (Body < Buffer.size()) {
    if (Buffer[Body] != '\r' && Buffer[Body] != '\n') {
      ++Body;
      continue;
    }
    if (Body > Decomposed.second && Buffer[Body - 1] == '\\') {
      if (Buffer[Body] == '\r' && Body + 1 < Buffer.size() &&
          Buffer[Body + 1] == '\n')
        ++Body;
      ++Body;
      continue;
    }
    break;
  }
  while (Body < Buffer.size() && (Buffer[Body] == '\r' || Buffer[Body] == '\n'))
    ++Body;

  const SourceLocation BodyLoc =
      SM.getLocForStartOfFile(Decomposed.first).getLocWithOffset(Body);
  std::string Text = Buffer.drop_front(Body).str();
  Lexer Lex(BodyLoc, PP.getLangOpts(), Text.data(), Text.data(),
            Text.data() + Text.size());
  unsigned Depth = 1;
  Token Tok;
  while (!Lex.LexFromRawLexer(Tok)) {
    if (Tok.isNot(tok::hash) || !Tok.isAtStartOfLine())
      continue;

    Token DirectiveTok;
    if (Lex.LexFromRawLexer(DirectiveTok))
      return false;
    StringRef Directive = getTokenName(DirectiveTok);
    if (Directive == "if" || Directive == "ifdef" || Directive == "ifndef") {
      ++Depth;
      continue;
    }
    if (Directive == "endif") {
      if (--Depth == 0)
        return false;
      continue;
    }
    if (Depth != 1)
      continue;
    if (Directive == "else" || Directive.starts_with("elif"))
      return false;
    if (Directive == "error")
      return true;
  }
  return false;
}

std::optional<std::string>
MacroConditionCallbacks::defaultedMacroInSkippedRange(SourceRange Range) const {
  const SourceLocation BeginLoc = SM.getExpansionLoc(Range.getBegin());
  const SourceLocation EndLoc = SM.getExpansionLoc(Range.getEnd());
  if (BeginLoc.isInvalid() || EndLoc.isInvalid())
    return std::nullopt;

  const std::pair<FileID, unsigned> Begin = SM.getDecomposedLoc(BeginLoc);
  const std::pair<FileID, unsigned> End = SM.getDecomposedLoc(EndLoc);
  if (Begin.first != End.first || Begin.second >= End.second)
    return std::nullopt;

  bool Invalid = false;
  StringRef Buffer = SM.getBufferData(Begin.first, &Invalid);
  if (Invalid || End.second > Buffer.size())
    return std::nullopt;

  std::string Text = Buffer.slice(Begin.second, End.second).str();
  Lexer Lex(BeginLoc, PP.getLangOpts(), Text.data(), Text.data(),
            Text.data() + Text.size());
  SmallVector<Token, 32> Tokens;
  Token Tok;
  bool AtEnd = false;
  do {
    AtEnd = Lex.LexFromRawLexer(Tok);
    if (Tok.isNot(tok::eof))
      Tokens.push_back(Tok);
  } while (!AtEnd);

  if (Tokens.size() < 3 || Tokens[0].isNot(tok::hash) ||
      !Tokens[0].isAtStartOfLine())
    return std::nullopt;

  size_t Index = 2;
  std::string Name;
  StringRef Directive = getTokenName(Tokens[1]);
  if (Directive == "ifndef") {
    if (!Tokens[Index].is(tok::raw_identifier))
      return std::nullopt;
    Name = Tokens[Index++].getRawIdentifier().str();
  } else if (Directive == "if") {
    if (Tokens[Index].isNot(tok::exclaim))
      return std::nullopt;
    ++Index;
    if (Index >= Tokens.size() || getTokenName(Tokens[Index]) != "defined")
      return std::nullopt;
    ++Index;
    if (Index < Tokens.size() && Tokens[Index].is(tok::l_paren))
      ++Index;
    if (Index >= Tokens.size() || Tokens[Index].isNot(tok::raw_identifier))
      return std::nullopt;
    Name = Tokens[Index++].getRawIdentifier().str();
    if (Index < Tokens.size() && Tokens[Index].is(tok::r_paren))
      ++Index;
  } else {
    return std::nullopt;
  }

  if (Index < Tokens.size() && !Tokens[Index].isAtStartOfLine())
    return std::nullopt;

  unsigned Depth = 1;
  for (; Index + 1 < Tokens.size(); ++Index) {
    if (Tokens[Index].isNot(tok::hash) || !Tokens[Index].isAtStartOfLine())
      continue;

    StringRef NestedDirective = getTokenName(Tokens[++Index]);
    if (NestedDirective == "if" || NestedDirective == "ifdef" ||
        NestedDirective == "ifndef") {
      ++Depth;
      continue;
    }
    if (NestedDirective == "endif") {
      if (--Depth == 0)
        break;
      continue;
    }
    if (Depth != 1)
      continue;
    if (NestedDirective == "else" || NestedDirective.starts_with("elif"))
      break;
    if (NestedDirective != "define" || Index + 2 >= Tokens.size() ||
        getTokenName(Tokens[Index + 1]) != Name ||
        Tokens[Index + 2].isAtStartOfLine())
      continue;

    const Token &MacroName = Tokens[Index + 1];
    const Token &Replacement = Tokens[Index + 2];
    const SourceLocation MacroNameEnd = Lexer::getLocForEndOfToken(
        MacroName.getLocation(), 0, SM, PP.getLangOpts());
    if (Replacement.is(tok::l_paren) &&
        MacroNameEnd == Replacement.getLocation())
      continue;
    return Name;
  }
  return std::nullopt;
}

MacroConditionCallbacks::MacroReference
MacroConditionCallbacks::referenceFromRange(SourceRange Range,
                                            SourceLocation Loc) const {
  ConditionReferences References = referencesInCondition(Range);
  if (!References.empty()) {
    References.front().Loc = Loc;
    References.front().Kind = ReferenceKind::Definition;
    return std::move(References.front());
  }
  return {{}, Loc, ReferenceKind::Definition};
}

bool MacroConditionCallbacks::isIgnoredIdentifier(StringRef Name) const {
  const IdentifierInfo *Info = PP.getIdentifierInfo(Name);
  return Name == "true" || Name == "false" ||
         Info->isCPlusPlusOperatorKeyword();
}

bool MacroConditionCallbacks::isEnclosedByDefinitionGuard(
    const MacroReference &Reference) const {
  const FileID File = SM.getFileID(SM.getSpellingLoc(Reference.Loc));
  for (const DefinitionGuards &Guards : reverse(ConditionalGuards)) {
    for (const DefinitionGuard &Guard : Guards)
      if (Guard.File == File && Guard.Name == Reference.Name)
        return true;
  }
  return false;
}

void MacroConditionCallbacks::beginConditional(DefinitionGuards Guards) {
  ConditionalGuards.push_back(std::move(Guards));
}

void MacroConditionCallbacks::changeConditional(DefinitionGuards Guards) {
  if (!ConditionalGuards.empty())
    ConditionalGuards.back() = std::move(Guards);
}

void MacroConditionCallbacks::processReferences(
    const ConditionReferences &References) {
  for (const MacroReference &Reference : References) {
    bool IsCompoundReference = false;
    for (const MacroReference &Other : References) {
      if (Reference.Name == Other.Name && Reference.Kind != Other.Kind) {
        IsCompoundReference = true;
        break;
      }
    }
    if (!IsCompoundReference)
      checkReference(Reference);
  }
}

void MacroConditionCallbacks::processDefinitionReference(StringRef Name,
                                                         SourceLocation Loc) {
  if (!Name.empty())
    checkReference({Name.str(), Loc, ReferenceKind::Definition});
}

void MacroConditionCallbacks::checkReference(const MacroReference &Reference) {
  if (Reference.Kind == ReferenceKind::NegatedDefinition ||
      isStandardPredefinedMacro(Reference.Name) ||
      (Reference.Kind == ReferenceKind::Value &&
       isEnclosedByDefinitionGuard(Reference)))
    return;

  const IdentifierInfo *Info = PP.getIdentifierInfo(Reference.Name);
  const MacroInfo *Macro = PP.getMacroDefinition(Info).getMacroInfo();
  if (!Macro || Macro->isBuiltinMacro() || Macro->isFunctionLike() ||
      Macro->tokens().empty())
    return;

  const SourceLocation SpellingLoc = SM.getSpellingLoc(Reference.Loc);
  if (SpellingLoc.isInvalid())
    return;

  MacroUsage &Usage = MacroUsages[Macro][SM.getFileID(SpellingLoc)];
  const bool IsDefinition = Reference.Kind == ReferenceKind::Definition;
  SourceLocation &CurrentLoc =
      IsDefinition ? Usage.DefinitionTestLoc : Usage.ValueTestLoc;
  const SourceLocation OtherLoc =
      IsDefinition ? Usage.ValueTestLoc : Usage.DefinitionTestLoc;
  if (CurrentLoc.isInvalid())
    CurrentLoc = Reference.Loc;
  if (Usage.Diagnosed || OtherLoc.isInvalid())
    return;

  const unsigned Kind = IsDefinition ? 0 : 1;
  Check->diag(Reference.Loc,
              "Macro '%0' checked here for %select{definition|value}1 after "
              "being checked for %select{value|definition}1")
      << Reference.Name << Kind;
  Check->diag(OtherLoc,
              "Macro '%0' first checked here for "
              "%select{value|definition}1",
              DiagnosticIDs::Note)
      << Reference.Name << Kind;
  Usage.Diagnosed = true;
}

void MacroConditionCallbacks::If(SourceLocation Loc, SourceRange ConditionRange,
                                 ConditionValueKind ConditionValue) {
  if (!branchUnconditionallyErrors(Loc))
    processReferences(referencesInCondition(ConditionRange));
  beginConditional(definitionGuardsInCondition(ConditionRange));
}

void MacroConditionCallbacks::Ifdef(SourceLocation Loc,
                                    const Token &MacroNameTok,
                                    const MacroDefinition &MD) {
  const StringRef Name = getTokenName(MacroNameTok);
  if (!branchUnconditionallyErrors(Loc))
    processDefinitionReference(Name, Loc);
  DefinitionGuards Guards;
  if (!Name.empty())
    Guards.push_back({Name.str(), SM.getFileID(SM.getSpellingLoc(
                                      MacroNameTok.getLocation()))});
  beginConditional(std::move(Guards));
}

void MacroConditionCallbacks::Ifndef(SourceLocation, const Token &,
                                     const MacroDefinition &) {
  beginConditional({});
}

void MacroConditionCallbacks::Elif(SourceLocation Loc,
                                   SourceRange ConditionRange,
                                   ConditionValueKind ConditionValue,
                                   SourceLocation IfLoc) {
  changeConditional({});
  if (!branchUnconditionallyErrors(Loc))
    processReferences(referencesInCondition(ConditionRange));
  changeConditional(definitionGuardsInCondition(ConditionRange));
}

void MacroConditionCallbacks::Elifdef(SourceLocation Loc,
                                      const Token &MacroNameTok,
                                      const MacroDefinition &MD) {
  changeConditional({});
  const StringRef Name = getTokenName(MacroNameTok);
  if (!branchUnconditionallyErrors(Loc))
    processDefinitionReference(Name, Loc);
  DefinitionGuards Guards;
  if (!Name.empty())
    Guards.push_back({Name.str(), SM.getFileID(SM.getSpellingLoc(
                                      MacroNameTok.getLocation()))});
  changeConditional(std::move(Guards));
}

void MacroConditionCallbacks::Elifdef(SourceLocation Loc,
                                      SourceRange ConditionRange,
                                      SourceLocation IfLoc) {
  changeConditional({});
  MacroReference Reference = referenceFromRange(ConditionRange, Loc);
  if (!Reference.Name.empty() && !branchUnconditionallyErrors(Loc))
    checkReference(Reference);
  DefinitionGuards Guards;
  if (!Reference.Name.empty())
    Guards.push_back({std::move(Reference.Name),
                      SM.getFileID(SM.getSpellingLoc(Reference.Loc))});
  changeConditional(std::move(Guards));
}

void MacroConditionCallbacks::Elifndef(SourceLocation, const Token &,
                                       const MacroDefinition &) {
  changeConditional({});
}

void MacroConditionCallbacks::Elifndef(SourceLocation, SourceRange,
                                       SourceLocation) {
  changeConditional({});
}

void MacroConditionCallbacks::Else(SourceLocation, SourceLocation) {
  changeConditional({});
}

void MacroConditionCallbacks::Endif(SourceLocation, SourceLocation) {
  if (!ConditionalGuards.empty())
    ConditionalGuards.pop_back();
}

void MacroConditionCallbacks::SourceRangeSkipped(SourceRange Range,
                                                 SourceLocation EndifLoc) {
  std::optional<std::string> Name = defaultedMacroInSkippedRange(Range);
  if (!Name)
    return;

  const IdentifierInfo *Info = PP.getIdentifierInfo(*Name);
  const MacroInfo *Macro = PP.getMacroDefinition(Info).getMacroInfo();
  const auto MacroIt = MacroUsages.find(Macro);
  if (MacroIt == MacroUsages.end())
    return;

  const SourceLocation GuardLoc = SM.getSpellingLoc(Range.getBegin());
  const FileID GuardFile = SM.getFileID(GuardLoc);
  const auto FileIt = MacroIt->second.find(GuardFile);
  if (FileIt == MacroIt->second.end())
    return;

  SourceLocation &DefinitionLoc = FileIt->second.DefinitionTestLoc;
  if (DefinitionLoc.isValid() && SM.getSpellingLineNumber(DefinitionLoc) ==
                                     SM.getSpellingLineNumber(GuardLoc))
    DefinitionLoc = {};
}

void MacroConditionCheck::registerPPCallbacks(const SourceManager &SM,
                                              Preprocessor *PP,
                                              Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<MacroConditionCallbacks>(this, SM, *PP));
}

} // namespace clang::tidy::bugprone
