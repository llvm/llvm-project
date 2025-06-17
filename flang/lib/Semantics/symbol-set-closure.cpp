//===-- lib/Semantics/symbol-set-closure.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/symbol-set-closure.h"
#include "flang/Common/idioms.h"
#include "flang/Common/visit.h"

namespace Fortran::semantics {

class Collector {
public:
  explicit Collector(int flags) : flags_{flags} {}

  UnorderedSymbolSet Collected() { return std::move(set_); }

  void operator()(const Symbol &x) { set_.insert(x); }
  void operator()(SymbolRef x) { (*this)(*x); }
  template <typename A> void operator()(const std::optional<A> &x) {
    if (x) {
      (*this)(*x);
    }
  }
  template <typename A> void operator()(const A *x) {
    if (x) {
      (*this)(*x);
    }
  }
  void operator()(const UnorderedSymbolSet &x) {
    for (const Symbol &symbol : x) {
      (*this)(symbol);
    }
  }
  void operator()(const SourceOrderedSymbolSet &x) {
    for (const Symbol &symbol : x) {
      (*this)(symbol);
    }
  }
  void operator()(const Scope &x) {
    for (const auto &[_, ref] : x) {
      (*this)(*ref);
    }
  }
  template <typename T> void operator()(const evaluate::Expr<T> &x) {
    UnorderedSymbolSet exprSyms{evaluate::CollectSymbols(x)};
    for (const Symbol &sym : exprSyms) {
      if (!sym.owner().IsDerivedType() || sym.has<DerivedTypeDetails>() ||
          (flags_ & IncludeComponentsInExprs)) {
        (*this)(sym);
      }
    }
  }
  void operator()(const DeclTypeSpec &type) {
    if (type.category() == DeclTypeSpec::Category::Character) {
      (*this)(type.characterTypeSpec().length());
    } else {
      (*this)(type.AsDerived());
    }
  }
  void operator()(const DerivedTypeSpec &type) {
    (*this)(type.originalTypeSymbol());
    for (const auto &[_, value] : type.parameters()) {
      (*this)(value);
    }
  }
  void operator()(const ParamValue &x) { (*this)(x.GetExplicit()); }
  void operator()(const Bound &x) { (*this)(x.GetExplicit()); }
  void operator()(const ShapeSpec &x) {
    (*this)(x.lbound());
    (*this)(x.ubound());
  }
  void operator()(const ArraySpec &x) {
    for (const ShapeSpec &shapeSpec : x) {
      (*this)(shapeSpec);
    }
  }

private:
  UnorderedSymbolSet set_;
  int flags_{NoDependenceCollectionFlags};
};

UnorderedSymbolSet CollectAllDependences(const Scope &scope, int flags) {
  UnorderedSymbolSet basis;
  for (const auto &[_, symbol] : scope) {
    basis.insert(*symbol);
  }
  return CollectAllDependences(basis, flags);
}

UnorderedSymbolSet CollectAllDependences(
    const UnorderedSymbolSet &original, int flags) {
  UnorderedSymbolSet result;
  if (flags & IncludeOriginalSymbols) {
    result = original;
  }
  UnorderedSymbolSet work{original};
  while (!work.empty()) {
    Collector collect{flags};
    for (const Symbol &symbol : work) {
      if (symbol.test(Symbol::Flag::CompilerCreated)) {
        continue;
      }
      collect(symbol.GetType());
      common::visit(
          common::visitors{
              [&collect, &symbol](const ObjectEntityDetails &x) {
                collect(x.shape());
                collect(x.coshape());
                if (IsNamedConstant(symbol) || symbol.owner().IsDerivedType()) {
                  collect(x.init());
                }
                collect(x.commonBlock());
                if (const auto *set{FindEquivalenceSet(symbol)}) {
                  for (const EquivalenceObject &equivObject : *set) {
                    collect(equivObject.symbol);
                  }
                }
              },
              [&collect, &symbol](const ProcEntityDetails &x) {
                collect(x.rawProcInterface());
                if (symbol.owner().IsDerivedType()) {
                  collect(x.init());
                }
                // TODO: worry about procedure pointers in common blocks?
              },
              [&collect](const ProcBindingDetails &x) { collect(x.symbol()); },
              [&collect](const SubprogramDetails &x) {
                for (const Symbol *dummy : x.dummyArgs()) {
                  collect(dummy);
                }
                if (x.isFunction()) {
                  collect(x.result());
                }
              },
              [&collect, &symbol](
                  const DerivedTypeDetails &) { collect(symbol.scope()); },
              [&collect, flags](const GenericDetails &x) {
                collect(x.derivedType());
                collect(x.specific());
                for (const Symbol &use : x.uses()) {
                  collect(use);
                }
                if (flags & IncludeSpecificsOfGenerics) {
                  for (const Symbol &specific : x.specificProcs()) {
                    collect(specific);
                  }
                }
              },
              [&collect](const NamelistDetails &x) {
                for (const Symbol &symbol : x.objects()) {
                  collect(symbol);
                }
              },
              [&collect](const CommonBlockDetails &x) {
                for (auto ref : x.objects()) {
                  collect(*ref);
                }
              },
              [&collect, &symbol, flags](const UseDetails &x) {
                if (flags & FollowUseAssociations) {
                  collect(x.symbol());
                }
              },
              [&collect](const HostAssocDetails &x) { collect(x.symbol()); },
              [](const auto &) {},
          },
          symbol.details());
    }
    work.clear();
    for (const Symbol &symbol : collect.Collected()) {
      if (result.find(symbol) == result.end() &&
          ((flags & IncludeOriginalSymbols) ||
              original.find(symbol) == original.end())) {
        result.insert(symbol);
        work.insert(symbol);
      }
    }
  }
  return result;
}

} // namespace Fortran::semantics
