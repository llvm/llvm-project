//===-- Lower/Support/Utils.cpp -- utilities --------------------*- C++ -*-===//
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

#include "flang/Lower/Support/Utils.h"

#include "flang/Common/indirection.h"
#include "flang/Lower/IterationSpace.h"
#include "flang/Semantics/tools.h"
#include <cstdint>
#include <optional>
#include <type_traits>

namespace Fortran::lower {
// Fortran::evaluate::Expr are functional values organized like an AST. A
// Fortran::evaluate::Expr is meant to be moved and cloned. Using the front end
// tools can often cause copies and extra wrapper classes to be added to any
// Fortran::evaluate::Expr. These values should not be assumed or relied upon to
// have an *object* identity. They are deeply recursive, irregular structures
// built from a large number of classes which do not use inheritance and
// necessitate a large volume of boilerplate code as a result.
//
// Contrastingly, LLVM data structures make ubiquitous assumptions about an
// object's identity via pointers to the object. An object's location in memory
// is thus very often an identifying relation.

// This class defines a hash computation of a Fortran::evaluate::Expr tree value
// so it can be used with llvm::DenseMap. The Fortran::evaluate::Expr need not
// have the same address.
class HashEvaluateExpr {
public:
  // A Se::Symbol is the only part of an Fortran::evaluate::Expr with an
  // identity property.
  static unsigned getHashValue(const Fortran::semantics::Symbol &x) {
    return static_cast<unsigned>(reinterpret_cast<std::intptr_t>(&x));
  }
  template <typename A, bool COPY>
  static unsigned getHashValue(const Fortran::common::Indirection<A, COPY> &x) {
    return getHashValue(x.value());
  }
  template <typename A>
  static unsigned getHashValue(const std::optional<A> &x) {
    if (x.has_value())
      return getHashValue(x.value());
    return 0u;
  }
  static unsigned getHashValue(const Fortran::evaluate::Subscript &x) {
    return Fortran::common::visit(
        [&](const auto &v) { return getHashValue(v); }, x.u);
  }
  static unsigned getHashValue(const Fortran::evaluate::Triplet &x) {
    return getHashValue(x.lower()) - getHashValue(x.upper()) * 5u -
           getHashValue(x.stride()) * 11u;
  }
  static unsigned getHashValue(const Fortran::evaluate::Component &x) {
    return getHashValue(x.base()) * 83u - getHashValue(x.GetLastSymbol());
  }
  static unsigned getHashValue(const Fortran::evaluate::ArrayRef &x) {
    unsigned subs = 1u;
    for (const Fortran::evaluate::Subscript &v : x.subscript())
      subs -= getHashValue(v);
    return getHashValue(x.base()) * 89u - subs;
  }
  static unsigned getHashValue(const Fortran::evaluate::CoarrayRef &x) {
    unsigned subs = 1u;
    for (const Fortran::evaluate::Subscript &v : x.subscript())
      subs -= getHashValue(v);
    unsigned cosubs = 3u;
    for (const Fortran::evaluate::Expr<Fortran::evaluate::SubscriptInteger> &v :
         x.cosubscript())
      cosubs -= getHashValue(v);
    unsigned syms = 7u;
    for (const Fortran::evaluate::SymbolRef &v : x.base())
      syms += getHashValue(v);
    return syms * 97u - subs - cosubs + getHashValue(x.stat()) + 257u +
           getHashValue(x.team());
  }
  static unsigned getHashValue(const Fortran::evaluate::NamedEntity &x) {
    if (x.IsSymbol())
      return getHashValue(x.GetFirstSymbol()) * 11u;
    return getHashValue(x.GetComponent()) * 13u;
  }
  static unsigned getHashValue(const Fortran::evaluate::DataRef &x) {
    return Fortran::common::visit(
        [&](const auto &v) { return getHashValue(v); }, x.u);
  }
  static unsigned getHashValue(const Fortran::evaluate::ComplexPart &x) {
    return getHashValue(x.complex()) - static_cast<unsigned>(x.part());
  }
  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  static unsigned getHashValue(
      const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>, TC2>
          &x) {
    return getHashValue(x.left()) - (static_cast<unsigned>(TC1) + 2u) -
           (static_cast<unsigned>(KIND) + 5u);
  }
  template <int KIND>
  static unsigned
  getHashValue(const Fortran::evaluate::ComplexComponent<KIND> &x) {
    return getHashValue(x.left()) -
           (static_cast<unsigned>(x.isImaginaryPart) + 1u) * 3u;
  }
  template <typename T>
  static unsigned getHashValue(const Fortran::evaluate::Parentheses<T> &x) {
    return getHashValue(x.left()) * 17u;
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Negate<Fortran::evaluate::Type<TC, KIND>> &x) {
    return getHashValue(x.left()) - (static_cast<unsigned>(TC) + 5u) -
           (static_cast<unsigned>(KIND) + 7u);
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Add<Fortran::evaluate::Type<TC, KIND>> &x) {
    return (getHashValue(x.left()) + getHashValue(x.right())) * 23u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND);
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Subtract<Fortran::evaluate::Type<TC, KIND>> &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 19u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND);
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Multiply<Fortran::evaluate::Type<TC, KIND>> &x) {
    return (getHashValue(x.left()) + getHashValue(x.right())) * 29u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND);
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Divide<Fortran::evaluate::Type<TC, KIND>> &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 31u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND);
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 37u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND);
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>> &x) {
    return (getHashValue(x.left()) + getHashValue(x.right())) * 41u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND) +
           static_cast<unsigned>(x.ordering) * 7u;
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 43u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND);
  }
  template <int KIND>
  static unsigned
  getHashValue(const Fortran::evaluate::ComplexConstructor<KIND> &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 47u +
           static_cast<unsigned>(KIND);
  }
  template <int KIND>
  static unsigned getHashValue(const Fortran::evaluate::Concat<KIND> &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 53u +
           static_cast<unsigned>(KIND);
  }
  template <int KIND>
  static unsigned getHashValue(const Fortran::evaluate::SetLength<KIND> &x) {
    return (getHashValue(x.left()) - getHashValue(x.right())) * 59u +
           static_cast<unsigned>(KIND);
  }
  static unsigned getHashValue(const Fortran::semantics::SymbolRef &sym) {
    return getHashValue(sym.get());
  }
  static unsigned getHashValue(const Fortran::evaluate::Substring &x) {
    return 61u *
               Fortran::common::visit(
                   [&](const auto &p) { return getHashValue(p); }, x.parent()) -
           getHashValue(x.lower()) - (getHashValue(x.lower()) + 1u);
  }
  static unsigned
  getHashValue(const Fortran::evaluate::StaticDataObject::Pointer &x) {
    return llvm::hash_value(x->name());
  }
  static unsigned getHashValue(const Fortran::evaluate::SpecificIntrinsic &x) {
    return llvm::hash_value(x.name);
  }
  template <typename A>
  static unsigned getHashValue(const Fortran::evaluate::Constant<A> &x) {
    // FIXME: Should hash the content.
    return 103u;
  }
  static unsigned getHashValue(const Fortran::evaluate::ActualArgument &x) {
    if (const Fortran::evaluate::Symbol *sym = x.GetAssumedTypeDummy())
      return getHashValue(*sym);
    return getHashValue(*x.UnwrapExpr());
  }
  static unsigned
  getHashValue(const Fortran::evaluate::ProcedureDesignator &x) {
    return Fortran::common::visit(
        [&](const auto &v) { return getHashValue(v); }, x.u);
  }
  static unsigned getHashValue(const Fortran::evaluate::ProcedureRef &x) {
    unsigned args = 13u;
    for (const std::optional<Fortran::evaluate::ActualArgument> &v :
         x.arguments())
      args -= getHashValue(v);
    return getHashValue(x.proc()) * 101u - args;
  }
  template <typename A>
  static unsigned
  getHashValue(const Fortran::evaluate::ArrayConstructor<A> &x) {
    // FIXME: hash the contents.
    return 127u;
  }
  static unsigned getHashValue(const Fortran::evaluate::ImpliedDoIndex &x) {
    return llvm::hash_value(toStringRef(x.name).str()) * 131u;
  }
  static unsigned getHashValue(const Fortran::evaluate::TypeParamInquiry &x) {
    return getHashValue(x.base()) * 137u - getHashValue(x.parameter()) * 3u;
  }
  static unsigned getHashValue(const Fortran::evaluate::DescriptorInquiry &x) {
    return getHashValue(x.base()) * 139u -
           static_cast<unsigned>(x.field()) * 13u +
           static_cast<unsigned>(x.dimension());
  }
  static unsigned
  getHashValue(const Fortran::evaluate::StructureConstructor &x) {
    // FIXME: hash the contents.
    return 149u;
  }
  template <int KIND>
  static unsigned getHashValue(const Fortran::evaluate::Not<KIND> &x) {
    return getHashValue(x.left()) * 61u + static_cast<unsigned>(KIND);
  }
  template <int KIND>
  static unsigned
  getHashValue(const Fortran::evaluate::LogicalOperation<KIND> &x) {
    unsigned result = getHashValue(x.left()) + getHashValue(x.right());
    return result * 67u + static_cast<unsigned>(x.logicalOperator) * 5u;
  }
  template <Fortran::common::TypeCategory TC, int KIND>
  static unsigned getHashValue(
      const Fortran::evaluate::Relational<Fortran::evaluate::Type<TC, KIND>>
          &x) {
    return (getHashValue(x.left()) + getHashValue(x.right())) * 71u +
           static_cast<unsigned>(TC) + static_cast<unsigned>(KIND) +
           static_cast<unsigned>(x.opr) * 11u;
  }
  template <typename A>
  static unsigned getHashValue(const Fortran::evaluate::Expr<A> &x) {
    return Fortran::common::visit(
        [&](const auto &v) { return getHashValue(v); }, x.u);
  }
  static unsigned getHashValue(
      const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x) {
    return Fortran::common::visit(
        [&](const auto &v) { return getHashValue(v); }, x.u);
  }
  template <typename A>
  static unsigned getHashValue(const Fortran::evaluate::Designator<A> &x) {
    return Fortran::common::visit(
        [&](const auto &v) { return getHashValue(v); }, x.u);
  }
  template <int BITS>
  static unsigned
  getHashValue(const Fortran::evaluate::value::Integer<BITS> &x) {
    return static_cast<unsigned>(x.ToSInt());
  }
  static unsigned getHashValue(const Fortran::evaluate::NullPointer &x) {
    return ~179u;
  }
};

// Define the is equals test for using Fortran::evaluate::Expr values with
// llvm::DenseMap.
class IsEqualEvaluateExpr {
public:
  // A Se::Symbol is the only part of an Fortran::evaluate::Expr with an
  // identity property.
  static bool isEqual(const Fortran::semantics::Symbol &x,
                      const Fortran::semantics::Symbol &y) {
    return isEqual(&x, &y);
  }
  static bool isEqual(const Fortran::semantics::Symbol *x,
                      const Fortran::semantics::Symbol *y) {
    return x == y;
  }
  template <typename A, bool COPY>
  static bool isEqual(const Fortran::common::Indirection<A, COPY> &x,
                      const Fortran::common::Indirection<A, COPY> &y) {
    return isEqual(x.value(), y.value());
  }
  template <typename A>
  static bool isEqual(const std::optional<A> &x, const std::optional<A> &y) {
    if (x.has_value() && y.has_value())
      return isEqual(x.value(), y.value());
    return !x.has_value() && !y.has_value();
  }
  template <typename A>
  static bool isEqual(const std::vector<A> &x, const std::vector<A> &y) {
    if (x.size() != y.size())
      return false;
    const std::size_t size = x.size();
    for (std::remove_const_t<decltype(size)> i = 0; i < size; ++i)
      if (!isEqual(x[i], y[i]))
        return false;
    return true;
  }
  static bool isEqual(const Fortran::evaluate::Subscript &x,
                      const Fortran::evaluate::Subscript &y) {
    return Fortran::common::visit(
        [&](const auto &v, const auto &w) { return isEqual(v, w); }, x.u, y.u);
  }
  static bool isEqual(const Fortran::evaluate::Triplet &x,
                      const Fortran::evaluate::Triplet &y) {
    return isEqual(x.lower(), y.lower()) && isEqual(x.upper(), y.upper()) &&
           isEqual(x.stride(), y.stride());
  }
  static bool isEqual(const Fortran::evaluate::Component &x,
                      const Fortran::evaluate::Component &y) {
    return isEqual(x.base(), y.base()) &&
           isEqual(x.GetLastSymbol(), y.GetLastSymbol());
  }
  static bool isEqual(const Fortran::evaluate::ArrayRef &x,
                      const Fortran::evaluate::ArrayRef &y) {
    return isEqual(x.base(), y.base()) && isEqual(x.subscript(), y.subscript());
  }
  static bool isEqual(const Fortran::evaluate::CoarrayRef &x,
                      const Fortran::evaluate::CoarrayRef &y) {
    return isEqual(x.base(), y.base()) &&
           isEqual(x.subscript(), y.subscript()) &&
           isEqual(x.cosubscript(), y.cosubscript()) &&
           isEqual(x.stat(), y.stat()) && isEqual(x.team(), y.team());
  }
  static bool isEqual(const Fortran::evaluate::NamedEntity &x,
                      const Fortran::evaluate::NamedEntity &y) {
    if (x.IsSymbol() && y.IsSymbol())
      return isEqual(x.GetFirstSymbol(), y.GetFirstSymbol());
    return !x.IsSymbol() && !y.IsSymbol() &&
           isEqual(x.GetComponent(), y.GetComponent());
  }
  static bool isEqual(const Fortran::evaluate::DataRef &x,
                      const Fortran::evaluate::DataRef &y) {
    return Fortran::common::visit(
        [&](const auto &v, const auto &w) { return isEqual(v, w); }, x.u, y.u);
  }
  static bool isEqual(const Fortran::evaluate::ComplexPart &x,
                      const Fortran::evaluate::ComplexPart &y) {
    return isEqual(x.complex(), y.complex()) && x.part() == y.part();
  }
  template <typename A, Fortran::common::TypeCategory TC2>
  static bool isEqual(const Fortran::evaluate::Convert<A, TC2> &x,
                      const Fortran::evaluate::Convert<A, TC2> &y) {
    return isEqual(x.left(), y.left());
  }
  template <int KIND>
  static bool isEqual(const Fortran::evaluate::ComplexComponent<KIND> &x,
                      const Fortran::evaluate::ComplexComponent<KIND> &y) {
    return isEqual(x.left(), y.left()) &&
           x.isImaginaryPart == y.isImaginaryPart;
  }
  template <typename T>
  static bool isEqual(const Fortran::evaluate::Parentheses<T> &x,
                      const Fortran::evaluate::Parentheses<T> &y) {
    return isEqual(x.left(), y.left());
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Negate<A> &x,
                      const Fortran::evaluate::Negate<A> &y) {
    return isEqual(x.left(), y.left());
  }
  template <typename A>
  static bool isBinaryEqual(const A &x, const A &y) {
    return isEqual(x.left(), y.left()) && isEqual(x.right(), y.right());
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Add<A> &x,
                      const Fortran::evaluate::Add<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Subtract<A> &x,
                      const Fortran::evaluate::Subtract<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Multiply<A> &x,
                      const Fortran::evaluate::Multiply<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Divide<A> &x,
                      const Fortran::evaluate::Divide<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Power<A> &x,
                      const Fortran::evaluate::Power<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Extremum<A> &x,
                      const Fortran::evaluate::Extremum<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::RealToIntPower<A> &x,
                      const Fortran::evaluate::RealToIntPower<A> &y) {
    return isBinaryEqual(x, y);
  }
  template <int KIND>
  static bool isEqual(const Fortran::evaluate::ComplexConstructor<KIND> &x,
                      const Fortran::evaluate::ComplexConstructor<KIND> &y) {
    return isBinaryEqual(x, y);
  }
  template <int KIND>
  static bool isEqual(const Fortran::evaluate::Concat<KIND> &x,
                      const Fortran::evaluate::Concat<KIND> &y) {
    return isBinaryEqual(x, y);
  }
  template <int KIND>
  static bool isEqual(const Fortran::evaluate::SetLength<KIND> &x,
                      const Fortran::evaluate::SetLength<KIND> &y) {
    return isBinaryEqual(x, y);
  }
  static bool isEqual(const Fortran::semantics::SymbolRef &x,
                      const Fortran::semantics::SymbolRef &y) {
    return isEqual(x.get(), y.get());
  }
  static bool isEqual(const Fortran::evaluate::Substring &x,
                      const Fortran::evaluate::Substring &y) {
    return Fortran::common::visit(
               [&](const auto &p, const auto &q) { return isEqual(p, q); },
               x.parent(), y.parent()) &&
           isEqual(x.lower(), y.lower()) && isEqual(x.upper(), y.upper());
  }
  static bool isEqual(const Fortran::evaluate::StaticDataObject::Pointer &x,
                      const Fortran::evaluate::StaticDataObject::Pointer &y) {
    return x->name() == y->name();
  }
  static bool isEqual(const Fortran::evaluate::SpecificIntrinsic &x,
                      const Fortran::evaluate::SpecificIntrinsic &y) {
    return x.name == y.name;
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Constant<A> &x,
                      const Fortran::evaluate::Constant<A> &y) {
    return x == y;
  }
  static bool isEqual(const Fortran::evaluate::ActualArgument &x,
                      const Fortran::evaluate::ActualArgument &y) {
    if (const Fortran::evaluate::Symbol *xs = x.GetAssumedTypeDummy()) {
      if (const Fortran::evaluate::Symbol *ys = y.GetAssumedTypeDummy())
        return isEqual(*xs, *ys);
      return false;
    }
    return !y.GetAssumedTypeDummy() &&
           isEqual(*x.UnwrapExpr(), *y.UnwrapExpr());
  }
  static bool isEqual(const Fortran::evaluate::ProcedureDesignator &x,
                      const Fortran::evaluate::ProcedureDesignator &y) {
    return Fortran::common::visit(
        [&](const auto &v, const auto &w) { return isEqual(v, w); }, x.u, y.u);
  }
  static bool isEqual(const Fortran::evaluate::ProcedureRef &x,
                      const Fortran::evaluate::ProcedureRef &y) {
    return isEqual(x.proc(), y.proc()) && isEqual(x.arguments(), y.arguments());
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::ArrayConstructor<A> &x,
                      const Fortran::evaluate::ArrayConstructor<A> &y) {
    llvm::report_fatal_error("not implemented");
  }
  static bool isEqual(const Fortran::evaluate::ImpliedDoIndex &x,
                      const Fortran::evaluate::ImpliedDoIndex &y) {
    return toStringRef(x.name) == toStringRef(y.name);
  }
  static bool isEqual(const Fortran::evaluate::TypeParamInquiry &x,
                      const Fortran::evaluate::TypeParamInquiry &y) {
    return isEqual(x.base(), y.base()) && isEqual(x.parameter(), y.parameter());
  }
  static bool isEqual(const Fortran::evaluate::DescriptorInquiry &x,
                      const Fortran::evaluate::DescriptorInquiry &y) {
    return isEqual(x.base(), y.base()) && x.field() == y.field() &&
           x.dimension() == y.dimension();
  }
  static bool isEqual(const Fortran::evaluate::StructureConstructor &x,
                      const Fortran::evaluate::StructureConstructor &y) {
    const auto &xValues = x.values();
    const auto &yValues = y.values();
    if (xValues.size() != yValues.size())
      return false;
    if (x.derivedTypeSpec() != y.derivedTypeSpec())
      return false;
    for (const auto &[xSymbol, xValue] : xValues) {
      auto yIt = yValues.find(xSymbol);
      // This should probably never happen, since the derived type
      // should be the same.
      if (yIt == yValues.end())
        return false;
      if (!isEqual(xValue, yIt->second))
        return false;
    }
    return true;
  }
  template <int KIND>
  static bool isEqual(const Fortran::evaluate::Not<KIND> &x,
                      const Fortran::evaluate::Not<KIND> &y) {
    return isEqual(x.left(), y.left());
  }
  template <int KIND>
  static bool isEqual(const Fortran::evaluate::LogicalOperation<KIND> &x,
                      const Fortran::evaluate::LogicalOperation<KIND> &y) {
    return isEqual(x.left(), y.left()) && isEqual(x.right(), y.right());
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Relational<A> &x,
                      const Fortran::evaluate::Relational<A> &y) {
    return isEqual(x.left(), y.left()) && isEqual(x.right(), y.right());
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Expr<A> &x,
                      const Fortran::evaluate::Expr<A> &y) {
    return Fortran::common::visit(
        [&](const auto &v, const auto &w) { return isEqual(v, w); }, x.u, y.u);
  }
  static bool
  isEqual(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &x,
          const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &y) {
    return Fortran::common::visit(
        [&](const auto &v, const auto &w) { return isEqual(v, w); }, x.u, y.u);
  }
  template <typename A>
  static bool isEqual(const Fortran::evaluate::Designator<A> &x,
                      const Fortran::evaluate::Designator<A> &y) {
    return Fortran::common::visit(
        [&](const auto &v, const auto &w) { return isEqual(v, w); }, x.u, y.u);
  }
  template <int BITS>
  static bool isEqual(const Fortran::evaluate::value::Integer<BITS> &x,
                      const Fortran::evaluate::value::Integer<BITS> &y) {
    return x == y;
  }
  static bool isEqual(const Fortran::evaluate::NullPointer &x,
                      const Fortran::evaluate::NullPointer &y) {
    return true;
  }
  template <typename A, typename B,
            std::enable_if_t<!std::is_same_v<A, B>, bool> = true>
  static bool isEqual(const A &, const B &) {
    return false;
  }
};

unsigned getHashValue(const Fortran::lower::SomeExpr *x) {
  return HashEvaluateExpr::getHashValue(*x);
}

unsigned getHashValue(const Fortran::lower::ExplicitIterSpace::ArrayBases &x) {
  return Fortran::common::visit(
      [&](const auto *p) { return HashEvaluateExpr::getHashValue(*p); }, x);
}

bool isEqual(const Fortran::lower::SomeExpr *x,
             const Fortran::lower::SomeExpr *y) {
  const auto *empty =
      llvm::DenseMapInfo<const Fortran::lower::SomeExpr *>::getEmptyKey();
  const auto *tombstone =
      llvm::DenseMapInfo<const Fortran::lower::SomeExpr *>::getTombstoneKey();
  if (x == empty || y == empty || x == tombstone || y == tombstone)
    return x == y;
  return x == y || IsEqualEvaluateExpr::isEqual(*x, *y);
}

bool isEqual(const Fortran::lower::ExplicitIterSpace::ArrayBases &x,
             const Fortran::lower::ExplicitIterSpace::ArrayBases &y) {
  return Fortran::common::visit(
      Fortran::common::visitors{
          // Fortran::semantics::Symbol * are the exception here. These pointers
          // have identity; if two Symbol * values are the same (different) then
          // they are the same (different) logical symbol.
          [&](Fortran::lower::FrontEndSymbol p,
              Fortran::lower::FrontEndSymbol q) { return p == q; },
          [&](const auto *p, const auto *q) {
            if constexpr (std::is_same_v<decltype(p), decltype(q)>) {
              return IsEqualEvaluateExpr::isEqual(*p, *q);
            } else {
              // Different subtree types are never equal.
              return false;
            }
          }},
      x, y);
}
} // end namespace Fortran::lower
