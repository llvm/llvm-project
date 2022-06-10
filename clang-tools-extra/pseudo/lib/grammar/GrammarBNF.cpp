//===--- GrammarBNF.cpp - build grammar from BNF files  ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/grammar/Grammar.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <utility>

namespace clang {
namespace pseudo {

namespace {
static const llvm::StringRef OptSuffix = "_opt";
static const llvm::StringRef StartSymbol = "_";

// Builds grammar from BNF files.
class GrammarBuilder {
public:
  GrammarBuilder(std::vector<std::string> &Diagnostics)
      : Diagnostics(Diagnostics) {}

  std::unique_ptr<Grammar> build(llvm::StringRef BNF) {
    auto Specs = eliminateOptional(parse(BNF));

    assert(llvm::all_of(Specs,
                        [](const RuleSpec &R) {
                          if (R.Target.endswith(OptSuffix))
                            return false;
                          return llvm::all_of(
                              R.Sequence, [](const RuleSpec::Element &E) {
                                return !E.Symbol.endswith(OptSuffix);
                              });
                        }) &&
           "Optional symbols should be eliminated!");

    auto T = std::make_unique<GrammarTable>();

    // Assemble the name->ID and ID->nonterminal name maps.
    llvm::DenseSet<llvm::StringRef> UniqueNonterminals;
    llvm::DenseMap<llvm::StringRef, SymbolID> SymbolIds;

    llvm::DenseSet<llvm::StringRef> UniqueAttributeValues;

    for (uint16_t I = 0; I < NumTerminals; ++I)
      SymbolIds.try_emplace(T->Terminals[I], tokenSymbol(tok::TokenKind(I)));
    auto Consider = [&](llvm::StringRef Name) {
      if (!SymbolIds.count(Name))
        UniqueNonterminals.insert(Name);
    };
    for (const auto &Spec : Specs) {
      Consider(Spec.Target);
      for (const RuleSpec::Element &Elt : Spec.Sequence) {
        Consider(Elt.Symbol);
        for (const auto& KV : Elt.Attributes)
           UniqueAttributeValues.insert(KV.second);
      }
    }
    llvm::for_each(UniqueNonterminals, [&T](llvm::StringRef Name) {
      T->Nonterminals.emplace_back();
      T->Nonterminals.back().Name = Name.str();
    });
    assert(T->Nonterminals.size() < (1 << (SymbolBits - 1)) &&
           "Too many nonterminals to fit in SymbolID bits!");
    llvm::sort(T->Nonterminals, [](const GrammarTable::Nonterminal &L,
                                   const GrammarTable::Nonterminal &R) {
      return L.Name < R.Name;
    });
    // Add an empty string for the corresponding sentinel unset attribute.
    T->AttributeValues.push_back("");
    llvm::for_each(UniqueAttributeValues, [&T](llvm::StringRef Name) {
      T->AttributeValues.emplace_back();
      T->AttributeValues.back() = Name.str();
    });
    llvm::sort(T->AttributeValues);
    assert(T->AttributeValues.front() == "");

    // Build name -> ID maps for nonterminals.
    for (SymbolID SID = 0; SID < T->Nonterminals.size(); ++SID)
      SymbolIds.try_emplace(T->Nonterminals[SID].Name, SID);

    // Convert the rules.
    T->Rules.reserve(Specs.size());
    std::vector<SymbolID> Symbols;
    auto Lookup = [SymbolIds](llvm::StringRef Name) {
      auto It = SymbolIds.find(Name);
      assert(It != SymbolIds.end() && "Didn't find the symbol in SymbolIds!");
      return It->second;
    };
    for (const auto &Spec : Specs) {
      assert(Spec.Sequence.size() <= Rule::MaxElements);
      Symbols.clear();
      for (const RuleSpec::Element &Elt : Spec.Sequence)
        Symbols.push_back(Lookup(Elt.Symbol));
      T->Rules.push_back(Rule(Lookup(Spec.Target), Symbols));
      applyAttributes(Spec, *T, T->Rules.back());
    }

    assert(T->Rules.size() < (1 << RuleBits) &&
           "Too many rules to fit in RuleID bits!");
    const auto &SymbolOrder = getTopologicalOrder(T.get());
    llvm::stable_sort(
        T->Rules, [&SymbolOrder](const Rule &Left, const Rule &Right) {
          // Sorted by the topological order of the nonterminal Target.
          return SymbolOrder[Left.Target] < SymbolOrder[Right.Target];
        });
    for (SymbolID SID = 0; SID < T->Nonterminals.size(); ++SID) {
      auto StartIt = llvm::partition_point(T->Rules, [&](const Rule &R) {
        return SymbolOrder[R.Target] < SymbolOrder[SID];
      });
      RuleID Start = StartIt - T->Rules.begin();
      RuleID End = Start;
      while (End < T->Rules.size() && T->Rules[End].Target == SID)
        ++End;
      T->Nonterminals[SID].RuleRange = {Start, End};
    }
    auto G = std::make_unique<Grammar>(std::move(T));
    diagnoseGrammar(*G);
    return G;
  }

  // Gets topological order for nonterminal symbols.
  //
  // The topological order is defined as: if a *single* nonterminal A produces
  // (or transitively) a nonterminal B (that said, there is a production rule
  // B := A), then A is less than B.
  //
  // It returns the sort key for each symbol, the array is indexed by SymbolID.
  std::vector<unsigned> getTopologicalOrder(GrammarTable *T) {
    std::vector<std::pair<SymbolID, SymbolID>> Dependencies;
    for (const auto &Rule : T->Rules) {
      // if A := B, A depends on B.
      if (Rule.Size == 1 && pseudo::isNonterminal(Rule.Sequence[0]))
        Dependencies.push_back({Rule.Target, Rule.Sequence[0]});
    }
    llvm::sort(Dependencies);
    std::vector<SymbolID> Order;
    // Each nonterminal state flows: NotVisited -> Visiting -> Visited.
    enum State {
      NotVisited,
      Visiting,
      Visited,
    };
    std::vector<State> VisitStates(T->Nonterminals.size(), NotVisited);
    std::function<void(SymbolID)> DFS = [&](SymbolID SID) -> void {
      if (VisitStates[SID] == Visited)
        return;
      if (VisitStates[SID] == Visiting) {
        Diagnostics.push_back(
            llvm::formatv("The grammar contains a cycle involving symbol {0}",
                          T->Nonterminals[SID].Name));
        return;
      }
      VisitStates[SID] = Visiting;
      for (auto It = llvm::lower_bound(Dependencies,
                                       std::pair<SymbolID, SymbolID>{SID, 0});
           It != Dependencies.end() && It->first == SID; ++It)
        DFS(It->second);
      VisitStates[SID] = Visited;
      Order.push_back(SID);
    };
    for (SymbolID ID = 0; ID != T->Nonterminals.size(); ++ID)
      DFS(ID);
    std::vector<unsigned> Result(T->Nonterminals.size(), 0);
    for (size_t I = 0; I < Order.size(); ++I)
      Result[Order[I]] = I;
    return Result;
  }

private:
  // Text representation of a BNF grammar rule.
  struct RuleSpec {
    llvm::StringRef Target;
    struct Element {
      llvm::StringRef Symbol; // Name of the symbol
      // Attributes that are associated to the sequence symbol or rule.
      std::vector<std::pair<llvm::StringRef/*Key*/, llvm::StringRef/*Value*/>>
          Attributes;
    };
    std::vector<Element> Sequence;

    std::string toString() const {
      std::vector<llvm::StringRef> Body;
      for (const auto &E : Sequence)
        Body.push_back(E.Symbol);
      return llvm::formatv("{0} := {1}", Target, llvm::join(Body, " "));
    }
  };

  std::vector<RuleSpec> parse(llvm::StringRef Lines) {
    std::vector<RuleSpec> Specs;
    for (llvm::StringRef Line : llvm::split(Lines, '\n')) {
      Line = Line.trim();
      // Strip anything coming after the '#' (comment).
      Line = Line.take_while([](char C) { return C != '#'; });
      if (Line.empty())
        continue;
      RuleSpec Rule;
      if (parseLine(Line, Rule))
        Specs.push_back(std::move(Rule));
    }
    return Specs;
  }

  bool parseLine(llvm::StringRef Line, RuleSpec &Out) {
    auto Parts = Line.split(":=");
    if (Parts.first == Line) { // no separator in Line
      Diagnostics.push_back(
          llvm::formatv("Failed to parse '{0}': no separator :=", Line).str());
      return false;
    }

    Out.Target = Parts.first.trim();
    Out.Sequence.clear();
    for (llvm::StringRef Chunk : llvm::split(Parts.second, ' ')) {
      Chunk = Chunk.trim();
      if (Chunk.empty())
        continue; // skip empty
      if (Chunk.startswith("[") && Chunk.endswith("]")) {
        if (Out.Sequence.empty())
          continue;

        parseAttributes(Chunk, Out.Sequence.back().Attributes);
        continue;
      }

      Out.Sequence.push_back({Chunk, /*Attributes=*/{}});
    }
    return true;
  }

  bool parseAttributes(
      llvm::StringRef Content,
      std::vector<std::pair<llvm::StringRef, llvm::StringRef>> &Out) {
    assert(Content.startswith("[") && Content.endswith("]"));
    auto KV = Content.drop_front().drop_back().split('=');
    Out.push_back({KV.first, KV.second.trim()});

    return true;
  }
  // Apply the parsed extensions (stored in RuleSpec) to the grammar Rule.
  void applyAttributes(const RuleSpec& Spec, const GrammarTable& T, Rule& R) {
    auto LookupExtensionID = [&T](llvm::StringRef Name) {
      const auto It = llvm::partition_point(
          T.AttributeValues, [&](llvm::StringRef X) { return X < Name; });
      assert(It != T.AttributeValues.end() && *It == Name &&
             "Didn't find the attribute in AttrValues!");
      return It - T.AttributeValues.begin();
    };
    for (const auto &KV : Spec.Sequence.back().Attributes) {
      if (KV.first == "guard") {
        R.Guard = LookupExtensionID(KV.second);
        continue;
      }
      Diagnostics.push_back(
          llvm::formatv("Unknown attribute '{0}'", KV.first).str());
    }
  }

  // Inlines all _opt symbols.
  // For example, a rule E := id +_opt id, after elimination, we have two
  // equivalent rules:
  //   1) E := id + id
  //   2) E := id id
  std::vector<RuleSpec> eliminateOptional(llvm::ArrayRef<RuleSpec> Input) {
    std::vector<RuleSpec> Results;
    std::vector<RuleSpec::Element> Storage;
    for (const auto &R : Input) {
      eliminateOptionalTail(
          R.Sequence, Storage, [&Results, &Storage, &R, this]() {
            if (Storage.empty()) {
              Diagnostics.push_back(
                  llvm::formatv("Rule '{0}' has a nullable RHS", R.toString()));
              return;
            }
            Results.push_back({R.Target, Storage});
          });
      assert(Storage.empty());
    }
    return Results;
  }
  void eliminateOptionalTail(llvm::ArrayRef<RuleSpec::Element> Elements,
                             std::vector<RuleSpec::Element> &Result,
                             llvm::function_ref<void()> CB) {
    if (Elements.empty())
      return CB();
    auto Front = Elements.front();
    if (!Front.Symbol.endswith(OptSuffix)) {
      Result.push_back(std::move(Front));
      eliminateOptionalTail(Elements.drop_front(1), Result, CB);
      Result.pop_back();
      return;
    }
    // Enumerate two options: skip the opt symbol, or inline the symbol.
    eliminateOptionalTail(Elements.drop_front(1), Result, CB); // skip
    Front.Symbol = Front.Symbol.drop_back(OptSuffix.size());   // drop "_opt"
    Result.push_back(std::move(Front));
    eliminateOptionalTail(Elements.drop_front(1), Result, CB);
    Result.pop_back();
  }

  // Diagnoses the grammar and emit warnings if any.
  void diagnoseGrammar(const Grammar &G) {
    const auto &T = G.table();
    for (SymbolID SID = 0; SID < T.Nonterminals.size(); ++SID) {
      auto Range = T.Nonterminals[SID].RuleRange;
      if (Range.Start == Range.End)
        Diagnostics.push_back(
            llvm::formatv("No rules for nonterminal: {0}", G.symbolName(SID)));
      llvm::StringRef NameRef = T.Nonterminals[SID].Name;
      if (llvm::all_of(NameRef, llvm::isAlpha) && NameRef.upper() == NameRef) {
        Diagnostics.push_back(llvm::formatv(
            "Token-like name {0} is used as a nonterminal", G.symbolName(SID)));
      }
    }
    for (RuleID RID = 0; RID + 1u < T.Rules.size(); ++RID) {
      if (T.Rules[RID] == T.Rules[RID + 1])
        Diagnostics.push_back(
            llvm::formatv("Duplicate rule: `{0}`", G.dumpRule(RID)));
    }
    // symbol-id -> used counts
    std::vector<unsigned> UseCounts(T.Nonterminals.size(), 0);
    for (const Rule &R : T.Rules)
      for (SymbolID SID : R.seq())
        if (isNonterminal(SID))
          ++UseCounts[SID];
    for (SymbolID SID = 0; SID < UseCounts.size(); ++SID)
      if (UseCounts[SID] == 0 && T.Nonterminals[SID].Name != StartSymbol)
        Diagnostics.push_back(
            llvm::formatv("Nonterminal never used: {0}", G.symbolName(SID)));
  }
  std::vector<std::string> &Diagnostics;
};
} // namespace

std::unique_ptr<Grammar>
Grammar::parseBNF(llvm::StringRef BNF, std::vector<std::string> &Diagnostics) {
  Diagnostics.clear();
  return GrammarBuilder(Diagnostics).build(BNF);
}

} // namespace pseudo
} // namespace clang
