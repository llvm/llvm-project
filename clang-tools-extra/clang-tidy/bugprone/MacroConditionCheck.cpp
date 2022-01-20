//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MacroConditionCheck.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include <memory>
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

private:
  struct MacroReference {
    std::string Name;
    SourceLocation Loc;
  };

  struct ConditionReferences {
    SmallVector<MacroReference, 2> Definition;
    SmallVector<MacroReference, 2> Value;
  };

  struct DefinitionCheck {
    std::string Name;
    SourceLocation DefinitionLoc;
    SourceLocation CheckLoc;
    bool ValueTested = false;
  };

  struct ConditionalBranch {
    SmallVector<DefinitionCheck, 2> Checks;
    llvm::StringSet<> DefinitionTests;
  };

  ConditionReferences referencesInCondition(SourceRange ConditionRange) const;
  MacroReference referenceFromRange(SourceRange Range,
                                    SourceLocation Loc) const;
  bool isIgnoredIdentifier(StringRef Name) const;
  void startCondition(const ConditionReferences &References);
  void startDefinitionCondition(StringRef Name, SourceLocation Loc);
  void nextBranch(const ConditionReferences &References = {});
  void processReferences(const ConditionReferences &References);
  void finishBranch(ConditionalBranch &Branch);
  void checkDefinitionReference(const MacroReference &Reference,
                                ConditionalBranch &Branch);
  void checkValueReference(const MacroReference &Reference);
  bool isDefinitionTestActive(StringRef Name) const;

  SmallVector<ConditionalBranch, 8> Conditions;
  llvm::DenseSet<unsigned> DiagnosedDefinitions;
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

MacroConditionCallbacks::ConditionReferences
MacroConditionCallbacks::referencesInCondition(
    SourceRange ConditionRange) const {
  ConditionReferences References;
  const SourceLocation BeginLoc = SM.getExpansionLoc(ConditionRange.getBegin());
  if (BeginLoc.isInvalid())
    return References;

  const std::pair<FileID, unsigned> Decomposed = SM.getDecomposedLoc(BeginLoc);
  bool Invalid = false;
  StringRef Buffer = SM.getBufferData(Decomposed.first, &Invalid);
  if (Invalid || Decomposed.second >= Buffer.size())
    return References;

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

  std::string Text = Buffer.slice(Decomposed.second, End).str();
  Lexer Lex(BeginLoc, PP.getLangOpts(), Text.data(), Text.data(),
            Text.data() + Text.size());
  SmallVector<Token, 16> Tokens;
  Token Tok;
  bool AtEnd = false;
  do {
    AtEnd = Lex.LexFromRawLexer(Tok);
    if (Tok.isNot(tok::eof))
      Tokens.push_back(Tok);
  } while (!AtEnd);

  for (size_t Index = 0; Index < Tokens.size(); ++Index) {
    const Token &Current = Tokens[Index];
    if (!Current.is(tok::raw_identifier))
      continue;

    StringRef Name = Current.getRawIdentifier();
    if (Name != "defined") {
      if (!isIgnoredIdentifier(Name))
        References.Value.push_back({Name.str(), Current.getLocation()});
      continue;
    }

    const SourceLocation DefinedLoc = Current.getLocation();
    ++Index;
    if (Index < Tokens.size() && Tokens[Index].is(tok::l_paren))
      ++Index;
    if (Index < Tokens.size() && Tokens[Index].is(tok::raw_identifier))
      References.Definition.push_back(
          {Tokens[Index].getRawIdentifier().str(), DefinedLoc});
  }
  return References;
}

MacroConditionCallbacks::MacroReference
MacroConditionCallbacks::referenceFromRange(SourceRange Range,
                                            SourceLocation Loc) const {
  ConditionReferences References = referencesInCondition(Range);
  if (!References.Value.empty()) {
    References.Value.front().Loc = Loc;
    return std::move(References.Value.front());
  }
  return {{}, Loc};
}

bool MacroConditionCallbacks::isIgnoredIdentifier(StringRef Name) const {
  const IdentifierInfo *Info = PP.getIdentifierInfo(Name);
  return Name == "true" || Name == "false" ||
         Info->isCPlusPlusOperatorKeyword();
}

void MacroConditionCallbacks::startCondition(
    const ConditionReferences &References) {
  Conditions.emplace_back();
  processReferences(References);
}

void MacroConditionCallbacks::startDefinitionCondition(StringRef Name,
                                                       SourceLocation Loc) {
  ConditionReferences References;
  if (!Name.empty())
    References.Definition.push_back({Name.str(), Loc});
  startCondition(References);
}

void MacroConditionCallbacks::nextBranch(
    const ConditionReferences &References) {
  if (Conditions.empty())
    return;
  finishBranch(Conditions.back());
  Conditions.back().Checks.clear();
  Conditions.back().DefinitionTests.clear();
  processReferences(References);
}

void MacroConditionCallbacks::processReferences(
    const ConditionReferences &References) {
  if (Conditions.empty())
    return;

  ConditionalBranch &Branch = Conditions.back();
  for (const MacroReference &Reference : References.Definition)
    checkDefinitionReference(Reference, Branch);
  for (const MacroReference &Reference : References.Value)
    checkValueReference(Reference);
}

void MacroConditionCallbacks::checkDefinitionReference(
    const MacroReference &Reference, ConditionalBranch &Branch) {
  Branch.DefinitionTests.insert(Reference.Name);

  const IdentifierInfo *Info = PP.getIdentifierInfo(Reference.Name);
  const MacroDefinition Definition = PP.getMacroDefinition(Info);
  const MacroInfo *Macro = Definition.getMacroInfo();
  if (!Macro || Macro->isBuiltinMacro() || Macro->isFunctionLike() ||
      Macro->tokens().empty())
    return;

  const SourceLocation DefinitionLoc = Macro->getDefinitionLoc();
  if (DefinitionLoc.isInvalid() || SM.getFilename(DefinitionLoc).empty())
    return;

  Branch.Checks.push_back(
      {Reference.Name, DefinitionLoc, Reference.Loc, false});
}

void MacroConditionCallbacks::checkValueReference(
    const MacroReference &Reference) {
  const IdentifierInfo *Info = PP.getIdentifierInfo(Reference.Name);
  if (!PP.getMacroDefinition(Info) && !isDefinitionTestActive(Reference.Name))
    Check->diag(Reference.Loc, "Undefined macro '%0' checked here for value")
        << Reference.Name;

  for (ConditionalBranch &Branch : Conditions)
    for (DefinitionCheck &Definition : Branch.Checks)
      if (Definition.Name == Reference.Name)
        Definition.ValueTested = true;
}

bool MacroConditionCallbacks::isDefinitionTestActive(StringRef Name) const {
  return llvm::any_of(Conditions, [Name](const ConditionalBranch &Branch) {
    return Branch.DefinitionTests.contains(Name);
  });
}

void MacroConditionCallbacks::finishBranch(ConditionalBranch &Branch) {
  for (const DefinitionCheck &Definition : Branch.Checks) {
    if (Definition.ValueTested)
      continue;

    if (DiagnosedDefinitions.insert(Definition.DefinitionLoc.getRawEncoding())
            .second)
      Check->diag(Definition.DefinitionLoc,
                  "Macro '%0' defined here with a value and checked for "
                  "definition")
          << Definition.Name;
    Check->diag(Definition.CheckLoc,
                "Macro '%0' defined with a value and checked here for "
                "definition")
        << Definition.Name;
  }
}

void MacroConditionCallbacks::If(SourceLocation Loc, SourceRange ConditionRange,
                                 ConditionValueKind ConditionValue) {
  startCondition(referencesInCondition(ConditionRange));
}

void MacroConditionCallbacks::Ifdef(SourceLocation Loc,
                                    const Token &MacroNameTok,
                                    const MacroDefinition &MD) {
  startDefinitionCondition(getTokenName(MacroNameTok), Loc);
}

void MacroConditionCallbacks::Ifndef(SourceLocation Loc,
                                     const Token &MacroNameTok,
                                     const MacroDefinition &MD) {
  startDefinitionCondition(getTokenName(MacroNameTok), Loc);
}

void MacroConditionCallbacks::Elif(SourceLocation Loc,
                                   SourceRange ConditionRange,
                                   ConditionValueKind ConditionValue,
                                   SourceLocation IfLoc) {
  nextBranch(referencesInCondition(ConditionRange));
}

void MacroConditionCallbacks::Elifdef(SourceLocation Loc,
                                      const Token &MacroNameTok,
                                      const MacroDefinition &MD) {
  ConditionReferences References;
  References.Definition.push_back({getTokenName(MacroNameTok).str(), Loc});
  nextBranch(References);
}

void MacroConditionCallbacks::Elifdef(SourceLocation Loc,
                                      SourceRange ConditionRange,
                                      SourceLocation IfLoc) {
  MacroReference Reference = referenceFromRange(ConditionRange, Loc);
  ConditionReferences References;
  if (!Reference.Name.empty())
    References.Definition.push_back(std::move(Reference));
  nextBranch(References);
}

void MacroConditionCallbacks::Elifndef(SourceLocation Loc,
                                       const Token &MacroNameTok,
                                       const MacroDefinition &MD) {
  ConditionReferences References;
  References.Definition.push_back({getTokenName(MacroNameTok).str(), Loc});
  nextBranch(References);
}

void MacroConditionCallbacks::Elifndef(SourceLocation Loc,
                                       SourceRange ConditionRange,
                                       SourceLocation IfLoc) {
  MacroReference Reference = referenceFromRange(ConditionRange, Loc);
  ConditionReferences References;
  if (!Reference.Name.empty())
    References.Definition.push_back(std::move(Reference));
  nextBranch(References);
}

void MacroConditionCallbacks::Else(SourceLocation Loc, SourceLocation IfLoc) {
  nextBranch();
}

void MacroConditionCallbacks::Endif(SourceLocation Loc, SourceLocation IfLoc) {
  if (Conditions.empty())
    return;
  finishBranch(Conditions.back());
  Conditions.pop_back();
}

void MacroConditionCheck::registerPPCallbacks(const SourceManager &SM,
                                              Preprocessor *PP,
                                              Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(std::make_unique<MacroConditionCallbacks>(this, SM, *PP));
}

} // namespace clang::tidy::bugprone
