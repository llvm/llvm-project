//===--- Grammar.cpp - Grammar for clang pseudo parser  ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace syntax {
namespace pseudo {

Rule::Rule(SymbolID Target, llvm::ArrayRef<SymbolID> Sequence)
    : Target(Target), Size(static_cast<uint8_t>(Sequence.size())) {
  assert(Sequence.size() <= Rule::MaxElements);
  llvm::copy(Sequence, this->Sequence);
}

llvm::ArrayRef<Rule> Grammar::rulesFor(SymbolID SID) const {
  assert(isNonterminal(SID));
  const auto &R = T->Nonterminals[SID].RuleRange;
  assert(R.end <= T->Rules.size());
  return llvm::makeArrayRef(&T->Rules[R.start], R.end - R.start);
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

std::string Grammar::dumpRule(RuleID RID) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  const Rule &R = T->Rules[RID];
  OS << symbolName(R.Target) << " :=";
  for (SymbolID SID : R.seq())
    OS << " " << symbolName(SID);
  return Result;
}

std::string Grammar::dumpRules(SymbolID SID) const {
  assert(isNonterminal(SID));
  std::string Result;
  const auto &Range = T->Nonterminals[SID].RuleRange;
  for (RuleID RID = Range.start; RID < Range.end; ++RID)
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

} // namespace pseudo
} // namespace syntax
} // namespace clang
