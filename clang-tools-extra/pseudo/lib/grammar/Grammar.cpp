//===--- Grammar.cpp - Grammar for clang pseudoparser  -----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/grammar/Grammar.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace clang {
namespace pseudo {

Rule::Rule(SymbolID Target, llvm::ArrayRef<SymbolID> Sequence)
    : Target(Target), Size(static_cast<uint8_t>(Sequence.size())) {
  assert(Sequence.size() <= Rule::MaxElements);
  llvm::copy(Sequence, this->Sequence);
}

Grammar::Grammar(std::unique_ptr<GrammarTable> Table) : T(std::move(Table)) {
  Underscore = *findNonterminal("_");
}

llvm::ArrayRef<Rule> Grammar::rulesFor(SymbolID SID) const {
  assert(isNonterminal(SID));
  const auto &R = T->Nonterminals[SID].RuleRange;
  assert(R.End <= T->Rules.size());
  return llvm::ArrayRef(&T->Rules[R.Start], R.End - R.Start);
}

const Rule &Grammar::lookupRule(RuleID RID) const {
  assert(RID < T->Rules.size());
  return T->Rules[RID];
}

llvm::StringRef Grammar::symbolName(SymbolID SID) const {
  if (isToken(SID))
    return T->Terminals[symbolToToken(SID)];
  return T->Nonterminals[SID].Name;
}

std::optional<SymbolID> Grammar::findNonterminal(llvm::StringRef Name) const {
  auto It = llvm::partition_point(
      T->Nonterminals,
      [&](const GrammarTable::Nonterminal &X) { return X.Name < Name; });
  if (It != T->Nonterminals.end() && It->Name == Name)
    return It - T->Nonterminals.begin();
  return std::nullopt;
}

std::string Grammar::dumpRule(RuleID RID) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  const Rule &R = T->Rules[RID];
  OS << symbolName(R.Target) << " :=";
  for (unsigned I = 0; I < R.Size; ++I) {
    OS << " " << symbolName(R.Sequence[I]);
    if (R.RecoveryIndex == I)
      OS << " [recover=" << T->AttributeValues[R.Recovery] << "]";
  }
  if (R.Guarded)
    OS << " [guard]";
  return Result;
}

std::string Grammar::dumpRules(SymbolID SID) const {
  assert(isNonterminal(SID));
  std::string Result;
  const auto &Range = T->Nonterminals[SID].RuleRange;
  for (RuleID RID = Range.Start; RID < Range.End; ++RID)
    Result.append(dumpRule(RID)).push_back('\n');
  return Result;
}

std::string Grammar::dump() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "Nonterminals:\n";
  for (SymbolID SID = 0; SID < T->Nonterminals.size(); ++SID)
    OS << llvm::formatv("  {0} {1}\n", SID, symbolName(SID));
  OS << "Rules:\n";
  for (RuleID RID = 0; RID < T->Rules.size(); ++RID)
    OS << llvm::formatv("  {0} {1}\n", RID, dumpRule(RID));
  return OS.str();
}

std::vector<llvm::DenseSet<SymbolID>> firstSets(const Grammar &G) {
  std::vector<llvm::DenseSet<SymbolID>> FirstSets(
      G.table().Nonterminals.size());
  auto ExpandFirstSet = [&FirstSets](SymbolID Target, SymbolID First) {
    assert(isNonterminal(Target));
    if (isToken(First))
      return FirstSets[Target].insert(First).second;
    bool Changed = false;
    for (SymbolID SID : FirstSets[First])
      Changed |= FirstSets[Target].insert(SID).second;
    return Changed;
  };

  // A rule S := T ... implies elements in FIRST(S):
  //  - if T is a terminal, FIRST(S) contains T
  //  - if T is a nonterminal, FIRST(S) contains FIRST(T)
  // Since FIRST(T) may not have been fully computed yet, FIRST(S) itself may
  // end up being incomplete.
  // We iterate until we hit a fixed point.
  // (This isn't particularly efficient, but table building isn't on the
  // critical path).
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const auto &R : G.table().Rules)
      // We only need to consider the first element because symbols are
      // non-nullable.
      Changed |= ExpandFirstSet(R.Target, R.seq().front());
  }
  return FirstSets;
}

std::vector<llvm::DenseSet<SymbolID>> followSets(const Grammar &G) {
  auto FirstSets = firstSets(G);
  std::vector<llvm::DenseSet<SymbolID>> FollowSets(
      G.table().Nonterminals.size());
  // Expand the follow set of a nonterminal symbol Y by adding all from the
  // given symbol set.
  auto ExpandFollowSet = [&FollowSets](SymbolID Y,
                                       const llvm::DenseSet<SymbolID> &ToAdd) {
    assert(isNonterminal(Y));
    bool Changed = false;
    for (SymbolID F : ToAdd)
      Changed |= FollowSets[Y].insert(F).second;
    return Changed;
  };
  // Follow sets is computed based on the following 3 rules, the computation
  // is completed at a fixed point where there is no more new symbols can be
  // added to any of the follow sets.
  //
  // Rule 1: add endmarker to the FOLLOW(S), where S is the start symbol of the
  // augmented grammar, in our case it is '_'.
  FollowSets[G.underscore()].insert(tokenSymbol(tok::eof));
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const auto &R : G.table().Rules) {
      // Rule 2: for a rule X := ... Y Z, we add all symbols from FIRST(Z) to
      // FOLLOW(Y).
      for (size_t I = 0; I + 1 < R.seq().size(); ++I) {
        if (isToken(R.seq()[I]))
          continue;
        // We only need to consider the next symbol because symbols are
        // non-nullable.
        SymbolID Next = R.seq()[I + 1];
        if (isToken(Next))
          // First set for a terminal is itself.
          Changed |= ExpandFollowSet(R.seq()[I], {Next});
        else
          Changed |= ExpandFollowSet(R.seq()[I], FirstSets[Next]);
      }
      // Rule 3: for a rule X := ... Z, we add all symbols from FOLLOW(X) to
      // FOLLOW(Z).
      SymbolID Z = R.seq().back();
      if (isNonterminal(Z))
        Changed |= ExpandFollowSet(Z, FollowSets[R.Target]);
    }
  }
  return FollowSets;
}

static llvm::ArrayRef<std::string> getTerminalNames() {
  static const auto &TerminalNames = []() {
    auto TerminalNames = new std::string[NumTerminals];
#define PUNCTUATOR(Tok, Spelling) TerminalNames[tok::Tok] = Spelling;
#define KEYWORD(Keyword, Condition)                                            \
  TerminalNames[tok::kw_##Keyword] = llvm::StringRef(#Keyword).upper();
#define TOK(Tok) TerminalNames[tok::Tok] = llvm::StringRef(#Tok).upper();
#include "clang/Basic/TokenKinds.def"
    return llvm::ArrayRef(TerminalNames, NumTerminals);
  }();
  return TerminalNames;
}
GrammarTable::GrammarTable() : Terminals(getTerminalNames()) {}

} // namespace pseudo
} // namespace clang
