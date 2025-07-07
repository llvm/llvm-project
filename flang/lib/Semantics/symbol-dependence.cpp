//===-- lib/Semantics/symbol-dependence.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/symbol-dependence.h"
#include "flang/Common/idioms.h"
#include "flang/Common/restorer.h"
#include "flang/Common/visit.h"
#include <queue>

static constexpr bool EnableDebugging{false};

namespace Fortran::semantics {

// Helper class that collects all of the symbol dependences for a
// given symbol.
class Collector {
public:
  explicit Collector(int flags) : flags_{flags} {}

  void CollectSymbolDependences(const Symbol &);
  UnorderedSymbolSet MustFollowDependences() { return std::move(dependences_); }
  SymbolVector AllDependences() { return std::move(mentions_); }

private:
  // This symbol is depended upon and its declaration must precede
  // the symbol of interest.
  void MustFollow(const Symbol &x) {
    if (!possibleImports_ || !DoesScopeContain(possibleImports_, x)) {
      dependences_.insert(x);
    }
  }
  // This symbol is depended upon, but is not necessarily a dependence
  // that must precede the symbol of interest in the output of the
  // topological sort.
  void Need(const Symbol &x) {
    if (mentioned_.insert(x).second) {
      mentions_.emplace_back(x);
    }
  }
  void Need(const Symbol *x) {
    if (x) {
      Need(*x);
    }
  }

  // These overloads of Collect() are mutally recursive, so they're
  // packaged as member functions of a class.
  void Collect(const Symbol &x) {
    Need(x);
    const auto *subp{x.detailsIf<SubprogramDetails>()};
    if ((subp && subp->isInterface()) || IsDummy(x) ||
        x.has<CommonBlockDetails>() || x.has<NamelistDetails>()) {
      // can be forward-referenced
    } else {
      MustFollow(x);
    }
  }
  void Collect(SymbolRef x) { Collect(*x); }
  template <typename A> void Collect(const std::optional<A> &x) {
    if (x) {
      Collect(*x);
    }
  }
  template <typename A> void Collect(const A *x) {
    if (x) {
      Collect(*x);
    }
  }
  void Collect(const UnorderedSymbolSet &x) {
    for (const Symbol &symbol : x) {
      Collect(symbol);
    }
  }
  void Collect(const SourceOrderedSymbolSet &x) {
    for (const Symbol &symbol : x) {
      Collect(symbol);
    }
  }
  void Collect(const SymbolVector &x) {
    for (const Symbol &symbol : x) {
      Collect(symbol);
    }
  }
  void Collect(const Scope &x) { Collect(x.GetSymbols()); }
  template <typename T> void Collect(const evaluate::Expr<T> &x) {
    UnorderedSymbolSet exprSyms{evaluate::CollectSymbols(x)};
    for (const Symbol &sym : exprSyms) {
      if (!sym.owner().IsDerivedType()) {
        Collect(sym);
      }
    }
  }
  void Collect(const DeclTypeSpec &type) {
    if (type.category() == DeclTypeSpec::Category::Character) {
      Collect(type.characterTypeSpec().length());
    } else {
      Collect(type.AsDerived());
    }
  }
  void Collect(const DerivedTypeSpec &type) {
    const Symbol &typeSym{type.originalTypeSymbol()};
    if (!derivedTypeReferenceCanBeForward_ || !type.parameters().empty()) {
      MustFollow(typeSym);
    }
    Need(typeSym);
    for (const auto &[_, value] : type.parameters()) {
      Collect(value);
    }
  }
  void Collect(const ParamValue &x) { Collect(x.GetExplicit()); }
  void Collect(const Bound &x) { Collect(x.GetExplicit()); }
  void Collect(const ShapeSpec &x) {
    Collect(x.lbound());
    Collect(x.ubound());
  }
  void Collect(const ArraySpec &x) {
    for (const ShapeSpec &shapeSpec : x) {
      Collect(shapeSpec);
    }
  }

  UnorderedSymbolSet mentioned_, dependences_;
  SymbolVector mentions_;
  int flags_{NoDependenceCollectionFlags};
  bool derivedTypeReferenceCanBeForward_{false};
  const Scope *possibleImports_{nullptr};
};

void Collector::CollectSymbolDependences(const Symbol &symbol) {
  if (symbol.has<ProcBindingDetails>() || symbol.has<SubprogramDetails>()) {
    // type will be picked up later for the function result, if any
  } else if (symbol.has<UseDetails>() || symbol.has<UseErrorDetails>() ||
      symbol.has<HostAssocDetails>()) {
  } else if (IsAllocatableOrPointer(symbol) && symbol.owner().IsDerivedType()) {
    bool saveCanBeForward{derivedTypeReferenceCanBeForward_};
    derivedTypeReferenceCanBeForward_ = true;
    Collect(symbol.GetType());
    derivedTypeReferenceCanBeForward_ = saveCanBeForward;
  } else {
    Collect(symbol.GetType());
  }
  common::visit(
      common::visitors{
          [this, &symbol](const ObjectEntityDetails &x) {
            Collect(x.shape());
            Collect(x.coshape());
            if (IsNamedConstant(symbol) || symbol.owner().IsDerivedType()) {
              Collect(x.init());
            }
            Need(x.commonBlock());
            if (const auto *set{FindEquivalenceSet(symbol)}) {
              for (const EquivalenceObject &equivObject : *set) {
                Need(equivObject.symbol);
              }
            }
            if (symbol.owner().IsModule()) {
              if (const EquivalenceSet *equiv{FindEquivalenceSet(symbol)}) {
                for (const EquivalenceObject &eqObj : *equiv) {
                  Need(eqObj.symbol);
                }
              }
            }
          },
          [this, &symbol](const ProcEntityDetails &x) {
            Collect(x.rawProcInterface());
            if (symbol.owner().IsDerivedType()) {
              Collect(x.init());
            }
          },
          [this](const ProcBindingDetails &x) { Need(x.symbol()); },
          [this, &symbol](const SubprogramDetails &x) {
            // Note dummy arguments & result symbol without dependence, unless
            // the subprogram is an interface block that might need to IMPORT
            // a type.
            bool needImports{x.isInterface()};
            auto restorer{common::ScopedSet(
                possibleImports_, needImports ? symbol.scope() : nullptr)};
            for (const Symbol *dummy : x.dummyArgs()) {
              if (dummy) {
                Need(*dummy);
                if (needImports) {
                  CollectSymbolDependences(*dummy);
                }
              }
            }
            if (x.isFunction()) {
              Need(x.result());
              if (needImports) {
                CollectSymbolDependences(x.result());
              }
            }
          },
          [this, &symbol](const DerivedTypeDetails &x) {
            Collect(symbol.scope());
            for (const auto &[_, symbolRef] : x.finals()) {
              Need(*symbolRef);
            }
          },
          [this](const GenericDetails &x) {
            Collect(x.derivedType());
            Collect(x.specific());
            if (flags_ & IncludeUsesOfGenerics) {
              for (const Symbol &use : x.uses()) {
                Collect(use);
              }
            }
            if (flags_ & IncludeSpecificsOfGenerics) {
              for (const Symbol &specific : x.specificProcs()) {
                Collect(specific);
              }
            }
          },
          [this](const NamelistDetails &x) {
            for (const Symbol &symbol : x.objects()) {
              Collect(symbol);
            }
          },
          [this](const CommonBlockDetails &x) {
            for (auto ref : x.objects()) {
              Collect(*ref);
            }
          },
          [this](const UseDetails &x) {
            if (flags_ & FollowUseAssociations) {
              Need(x.symbol());
            }
          },
          [this](const HostAssocDetails &x) { Need(x.symbol()); },
          [](const auto &) {},
      },
      symbol.details());
}

SymbolVector CollectAllDependences(const Scope &scope, int flags) {
  SymbolVector basis{scope.GetSymbols()};
  return CollectAllDependences(basis, flags, &scope);
}

// Returns a vector of symbols, topologically sorted by dependence
SymbolVector CollectAllDependences(
    const SymbolVector &original, int flags, const Scope *forScope) {
  std::queue<const Symbol *> work;
  UnorderedSymbolSet enqueued;
  for (const Symbol &symbol : original) {
    if (!symbol.test(Symbol::Flag::CompilerCreated)) {
      work.push(&symbol);
      enqueued.insert(symbol);
    }
  }
  // For each symbol, collect its dependences into "topology".
  // The "visited" vector and "enqueued" set hold all of the
  // symbols considered.
  std::map<const Symbol *, UnorderedSymbolSet> topology;
  std::vector<const Symbol *> visited;
  visited.reserve(2 * original.size());
  std::optional<SourceName> forModuleName;
  if (forScope && !(flags & NotJustForOneModule)) {
    if (const Scope *forModule{FindModuleContaining(*forScope)}) {
      forModuleName = forModule->GetName();
    }
  }
  while (!work.empty()) {
    const Symbol &symbol{*work.front()};
    work.pop();
    visited.push_back(&symbol);
    Collector collector{flags};
    bool doCollection{true};
    if (forModuleName) {
      if (const Scope *symModule{FindModuleContaining(symbol.owner())}) {
        if (auto symModName{symModule->GetName()}) {
          doCollection = *forModuleName == *symModName;
        }
      }
    }
    if (doCollection) {
      collector.CollectSymbolDependences(symbol);
    }
    auto dependences{collector.MustFollowDependences()};
    auto mentions{collector.AllDependences()};
    if constexpr (EnableDebugging) {
      for (const Symbol &need : dependences) {
        llvm::errs() << "symbol " << symbol << " must follow " << need << '\n';
      }
      for (const Symbol &need : mentions) {
        llvm::errs() << "symbol " << symbol << " needs " << need << '\n';
      }
    }
    CHECK(topology.find(&symbol) == topology.end());
    topology.emplace(&symbol, std::move(dependences));
    for (const Symbol &symbol : mentions) {
      if (!symbol.test(Symbol::Flag::CompilerCreated)) {
        if (enqueued.insert(symbol).second) {
          work.push(&symbol);
        }
      }
    }
  }
  CHECK(enqueued.size() == visited.size());
  // Topological sorting
  // Subtle: This inverted topology map uses a SymbolVector, not a set
  // of symbols, so that the order of symbols in the final output remains
  // deterministic.
  std::map<const Symbol *, SymbolVector> invertedTopology;
  for (const Symbol *symbol : visited) {
    invertedTopology[symbol] = SymbolVector{};
  }
  std::map<const Symbol *, std::size_t> numWaitingFor;
  for (const Symbol *symbol : visited) {
    auto topoIter{topology.find(symbol)};
    CHECK(topoIter != topology.end());
    const auto &needs{topoIter->second};
    if (needs.empty()) {
      work.push(symbol);
    } else {
      numWaitingFor[symbol] = needs.size();
      for (const Symbol &need : needs) {
        invertedTopology[&need].push_back(*symbol);
      }
    }
  }
  CHECK(visited.size() == work.size() + numWaitingFor.size());
  SymbolVector resultVector;
  while (!work.empty()) {
    const Symbol &symbol{*work.front()};
    work.pop();
    resultVector.push_back(symbol);
    auto enqueuedIter{enqueued.find(symbol)};
    CHECK(enqueuedIter != enqueued.end());
    enqueued.erase(enqueuedIter);
    if (auto invertedIter{invertedTopology.find(&symbol)};
        invertedIter != invertedTopology.end()) {
      for (const Symbol &neededBy : invertedIter->second) {
        std::size_t stillAwaiting{numWaitingFor[&neededBy] - 1};
        if (stillAwaiting == 0) {
          work.push(&neededBy);
        } else {
          numWaitingFor[&neededBy] = stillAwaiting;
        }
      }
    }
  }
  if constexpr (EnableDebugging) {
    llvm::errs() << "Topological sort failed in CollectAllDependences\n";
    for (const Symbol &remnant : enqueued) {
      auto topoIter{topology.find(&remnant)};
      CHECK(topoIter != topology.end());
      llvm::errs() << "  remnant symbol " << remnant << " needs:\n";
      for (const Symbol &n : topoIter->second) {
        llvm::errs() << "   " << n << '\n';
      }
    }
  }
  CHECK(enqueued.empty());
  CHECK(resultVector.size() == visited.size());
  return resultVector;
}

} // namespace Fortran::semantics
