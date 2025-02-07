//===-- IterationSpace.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IterationSpace.h"
#include "flang/Evaluate/expression.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Support/Utils.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-iteration-space"

namespace {

/// This class can recover the base array in an expression that contains
/// explicit iteration space symbols. Most of the class can be ignored as it is
/// boilerplate Fortran::evaluate::Expr traversal.
class ArrayBaseFinder {
public:
  using RT = bool;

  ArrayBaseFinder(llvm::ArrayRef<Fortran::lower::FrontEndSymbol> syms)
      : controlVars(syms) {}

  template <typename T>
  void operator()(const T &x) {
    (void)find(x);
  }

  /// Get the list of bases.
  llvm::ArrayRef<Fortran::lower::ExplicitIterSpace::ArrayBases>
  getBases() const {
    LLVM_DEBUG(llvm::dbgs()
               << "number of array bases found: " << bases.size() << '\n');
    return bases;
  }

private:
  // First, the cases that are of interest.
  RT find(const Fortran::semantics::Symbol &symbol) {
    if (symbol.Rank() > 0) {
      bases.push_back(&symbol);
      return true;
    }
    return {};
  }
  RT find(const Fortran::evaluate::Component &x) {
    auto found = find(x.base());
    if (!found && x.base().Rank() == 0 && x.Rank() > 0) {
      bases.push_back(&x);
      return true;
    }
    return found;
  }
  RT find(const Fortran::evaluate::ArrayRef &x) {
    for (const auto &sub : x.subscript())
      (void)find(sub);
    if (x.base().IsSymbol()) {
      if (x.Rank() > 0 || intersection(x.subscript())) {
        bases.push_back(&x);
        return true;
      }
      return {};
    }
    auto found = find(x.base());
    if (!found && ((x.base().Rank() == 0 && x.Rank() > 0) ||
                   intersection(x.subscript()))) {
      bases.push_back(&x);
      return true;
    }
    return found;
  }
  RT find(const Fortran::evaluate::Triplet &x) {
    if (const auto *lower = x.GetLower())
      (void)find(*lower);
    if (const auto *upper = x.GetUpper())
      (void)find(*upper);
    return find(x.GetStride());
  }
  RT find(const Fortran::evaluate::IndirectSubscriptIntegerExpr &x) {
    return find(x.value());
  }
  RT find(const Fortran::evaluate::Subscript &x) { return find(x.u); }
  RT find(const Fortran::evaluate::DataRef &x) { return find(x.u); }
  RT find(const Fortran::evaluate::CoarrayRef &x) {
    assert(false && "coarray reference");
    return {};
  }

  template <typename A>
  bool intersection(const A &subscripts) {
    return Fortran::lower::symbolsIntersectSubscripts(controlVars, subscripts);
  }

  // The rest is traversal boilerplate and can be ignored.
  RT find(const Fortran::evaluate::Substring &x) { return find(x.parent()); }
  template <typename A>
  RT find(const Fortran::semantics::SymbolRef x) {
    return find(*x);
  }
  RT find(const Fortran::evaluate::NamedEntity &x) {
    if (x.IsSymbol())
      return find(x.GetFirstSymbol());
    return find(x.GetComponent());
  }

  template <typename A, bool C>
  RT find(const Fortran::common::Indirection<A, C> &x) {
    return find(x.value());
  }
  template <typename A>
  RT find(const std::unique_ptr<A> &x) {
    return find(x.get());
  }
  template <typename A>
  RT find(const std::shared_ptr<A> &x) {
    return find(x.get());
  }
  template <typename A>
  RT find(const A *x) {
    if (x)
      return find(*x);
    return {};
  }
  template <typename A>
  RT find(const std::optional<A> &x) {
    if (x)
      return find(*x);
    return {};
  }
  template <typename... A>
  RT find(const std::variant<A...> &u) {
    return Fortran::common::visit([&](const auto &v) { return find(v); }, u);
  }
  template <typename A>
  RT find(const std::vector<A> &x) {
    for (auto &v : x)
      (void)find(v);
    return {};
  }
  RT find(const Fortran::evaluate::BOZLiteralConstant &) { return {}; }
  RT find(const Fortran::evaluate::NullPointer &) { return {}; }
  template <typename T>
  RT find(const Fortran::evaluate::Constant<T> &x) {
    return {};
  }
  RT find(const Fortran::evaluate::StaticDataObject &) { return {}; }
  RT find(const Fortran::evaluate::ImpliedDoIndex &) { return {}; }
  RT find(const Fortran::evaluate::BaseObject &x) {
    (void)find(x.u);
    return {};
  }
  RT find(const Fortran::evaluate::TypeParamInquiry &) { return {}; }
  RT find(const Fortran::evaluate::ComplexPart &x) { return {}; }
  template <typename T>
  RT find(const Fortran::evaluate::Designator<T> &x) {
    return find(x.u);
  }
  template <typename T>
  RT find(const Fortran::evaluate::Variable<T> &x) {
    return find(x.u);
  }
  RT find(const Fortran::evaluate::DescriptorInquiry &) { return {}; }
  RT find(const Fortran::evaluate::SpecificIntrinsic &) { return {}; }
  RT find(const Fortran::evaluate::ProcedureDesignator &x) { return {}; }
  RT find(const Fortran::evaluate::ProcedureRef &x) {
    (void)find(x.proc());
    if (x.IsElemental())
      (void)find(x.arguments());
    return {};
  }
  RT find(const Fortran::evaluate::ActualArgument &x) {
    if (const auto *sym = x.GetAssumedTypeDummy())
      (void)find(*sym);
    else
      (void)find(x.UnwrapExpr());
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::FunctionRef<T> &x) {
    (void)find(static_cast<const Fortran::evaluate::ProcedureRef &>(x));
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::ArrayConstructorValue<T> &) {
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::ArrayConstructorValues<T> &) {
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::ImpliedDo<T> &) {
    return {};
  }
  RT find(const Fortran::semantics::ParamValue &) { return {}; }
  RT find(const Fortran::semantics::DerivedTypeSpec &) { return {}; }
  RT find(const Fortran::evaluate::StructureConstructor &) { return {}; }
  template <typename D, typename R, typename O>
  RT find(const Fortran::evaluate::Operation<D, R, O> &op) {
    (void)find(op.left());
    return false;
  }
  template <typename D, typename R, typename LO, typename RO>
  RT find(const Fortran::evaluate::Operation<D, R, LO, RO> &op) {
    (void)find(op.left());
    (void)find(op.right());
    return false;
  }
  RT find(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x) {
    (void)find(x.u);
    return {};
  }
  template <typename T>
  RT find(const Fortran::evaluate::Expr<T> &x) {
    (void)find(x.u);
    return {};
  }

  llvm::SmallVector<Fortran::lower::ExplicitIterSpace::ArrayBases> bases;
  llvm::SmallVector<Fortran::lower::FrontEndSymbol> controlVars;
};

} // namespace

void Fortran::lower::ExplicitIterSpace::leave() {
  ccLoopNest.pop_back();
  --forallContextOpen;
  conditionalCleanup();
}

void Fortran::lower::ExplicitIterSpace::addSymbol(
    Fortran::lower::FrontEndSymbol sym) {
  assert(!symbolStack.empty());
  symbolStack.back().push_back(sym);
}

void Fortran::lower::ExplicitIterSpace::exprBase(Fortran::lower::FrontEndExpr x,
                                                 bool lhs) {
  ArrayBaseFinder finder(collectAllSymbols());
  finder(*x);
  llvm::ArrayRef<Fortran::lower::ExplicitIterSpace::ArrayBases> bases =
      finder.getBases();
  if (rhsBases.empty())
    endAssign();
  if (lhs) {
    if (bases.empty()) {
      lhsBases.push_back(std::nullopt);
      return;
    }
    assert(bases.size() >= 1 && "must detect an array reference on lhs");
    if (bases.size() > 1)
      rhsBases.back().append(bases.begin(), bases.end() - 1);
    lhsBases.push_back(bases.back());
    return;
  }
  rhsBases.back().append(bases.begin(), bases.end());
}

void Fortran::lower::ExplicitIterSpace::endAssign() { rhsBases.emplace_back(); }

void Fortran::lower::ExplicitIterSpace::pushLevel() {
  symbolStack.push_back(llvm::SmallVector<Fortran::lower::FrontEndSymbol>{});
}

void Fortran::lower::ExplicitIterSpace::popLevel() { symbolStack.pop_back(); }

void Fortran::lower::ExplicitIterSpace::conditionalCleanup() {
  if (forallContextOpen == 0) {
    // Exiting the outermost FORALL context.
    // Cleanup any residual mask buffers.
    outermostContext().finalizeAndReset();
    // Clear and reset all the cached information.
    symbolStack.clear();
    lhsBases.clear();
    rhsBases.clear();
    loadBindings.clear();
    ccLoopNest.clear();
    innerArgs.clear();
    outerLoop = std::nullopt;
    clearLoops();
    counter = 0;
  }
}

std::optional<size_t>
Fortran::lower::ExplicitIterSpace::findArgPosition(fir::ArrayLoadOp load) {
  if (lhsBases[counter]) {
    auto ld = loadBindings.find(*lhsBases[counter]);
    std::optional<size_t> optPos;
    if (ld != loadBindings.end() && ld->second == load)
      optPos = static_cast<size_t>(0u);
    assert(optPos.has_value() && "load does not correspond to lhs");
    return optPos;
  }
  return std::nullopt;
}

llvm::SmallVector<Fortran::lower::FrontEndSymbol>
Fortran::lower::ExplicitIterSpace::collectAllSymbols() {
  llvm::SmallVector<Fortran::lower::FrontEndSymbol> result;
  for (llvm::SmallVector<FrontEndSymbol> vec : symbolStack)
    result.append(vec.begin(), vec.end());
  return result;
}

llvm::raw_ostream &
Fortran::lower::operator<<(llvm::raw_ostream &s,
                           const Fortran::lower::ImplicitIterSpace &e) {
  for (const llvm::SmallVector<
           Fortran::lower::ImplicitIterSpace::FrontEndMaskExpr> &xs :
       e.getMasks()) {
    s << "{ ";
    for (const Fortran::lower::ImplicitIterSpace::FrontEndMaskExpr &x : xs)
      x->AsFortran(s << '(') << "), ";
    s << "}\n";
  }
  return s;
}

llvm::raw_ostream &
Fortran::lower::operator<<(llvm::raw_ostream &s,
                           const Fortran::lower::ExplicitIterSpace &e) {
  auto dump = [&](const auto &u) {
    Fortran::common::visit(
        Fortran::common::visitors{
            [&](const Fortran::semantics::Symbol *y) {
              s << "  " << *y << '\n';
            },
            [&](const Fortran::evaluate::ArrayRef *y) {
              s << "  ";
              if (y->base().IsSymbol())
                s << y->base().GetFirstSymbol();
              else
                s << y->base().GetComponent().GetLastSymbol();
              s << '\n';
            },
            [&](const Fortran::evaluate::Component *y) {
              s << "  " << y->GetLastSymbol() << '\n';
            }},
        u);
  };
  s << "LHS bases:\n";
  for (const std::optional<Fortran::lower::ExplicitIterSpace::ArrayBases> &u :
       e.lhsBases)
    if (u)
      dump(*u);
  s << "RHS bases:\n";
  for (const llvm::SmallVector<Fortran::lower::ExplicitIterSpace::ArrayBases>
           &bases : e.rhsBases) {
    for (const Fortran::lower::ExplicitIterSpace::ArrayBases &u : bases)
      dump(u);
    s << '\n';
  }
  return s;
}

void Fortran::lower::ImplicitIterSpace::dump() const {
  llvm::errs() << *this << '\n';
}

void Fortran::lower::ExplicitIterSpace::dump() const {
  llvm::errs() << *this << '\n';
}
