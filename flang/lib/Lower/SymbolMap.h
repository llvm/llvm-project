//===-- SymbolMap.h -- lowering internal symbol map -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_SYMBOLMAP_H
#define FORTRAN_LOWER_SYMBOLMAP_H

#include "flang/Common/idioms.h"
#include "flang/Common/reference.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Semantics/symbol.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace Fortran::lower {

//===----------------------------------------------------------------------===//
// Symbol information
//===----------------------------------------------------------------------===//

/// A dictionary entry of ssa-values that together compose a variable referenced
/// by a Symbol. For example, the declaration
///
///   CHARACTER(LEN=i) :: c(j1,j2)
///
/// is a single variable `c`. This variable is a two-dimensional array of
/// CHARACTER. It has a starting address and three dynamic properties: the LEN
/// parameter `i` a runtime value describing the length of the CHARACTER, and
/// the `j1` and `j2` runtime values, which describe the shape of the array.
///
/// The lowering bridge needs to be able to record all four of these ssa-values
/// in the lookup table to be able to correctly lower Fortran to FIR.
struct SymbolBox {
  // For lookups that fail, have a monostate
  using None = std::monostate;

  // Trivial intrinsic type
  using Intrinsic = fir::AbstractBox;

  // Array variable that uses bounds notation
  using FullDim = fir::ArrayBoxValue;

  // CHARACTER type variable with its dependent type LEN parameter
  using Char = fir::CharBoxValue;

  // CHARACTER array variable using bounds notation
  using CharFullDim = fir::CharArrayBoxValue;

  // Generalized derived type variable
  using Derived = fir::BoxValue;

  //===--------------------------------------------------------------------===//
  // Constructors
  //===--------------------------------------------------------------------===//

  SymbolBox() : box{None{}} {}
  template <typename A>
  SymbolBox(const A &x) : box{x} {}

  operator bool() const { return !std::holds_alternative<None>(box); }

  // This operator returns the address of the boxed value. TODO: consider
  // eliminating this in favor of explicit conversion.
  operator mlir::Value() const { return getAddr(); }

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  /// Get address of the boxed value. For a scalar, this is the address of the
  /// scalar. For an array, this is the address of the first element in the
  /// array, etc.
  mlir::Value getAddr() const {
    return std::visit(common::visitors{
                          [](const None &) { return mlir::Value{}; },
                          [](const auto &x) { return x.addr; },
                      },
                      box);
  }

  /// Get the LEN type parameter of a CHARACTER boxed value.
  llvm::Optional<mlir::Value> getCharLen() const {
    using T = llvm::Optional<mlir::Value>;
    return std::visit(common::visitors{
                          [](const Char &x) { return T{x.len}; },
                          [](const CharFullDim &x) { return T{x.len}; },
                          [](const auto &) { return T{}; },
                      },
                      box);
  }

  /// Does the boxed value have an intrinsic type?
  bool isIntrinsic() const {
    return std::visit(common::visitors{
                          [](const Intrinsic &) { return true; },
                          [](const Char &) { return true; },
                          [](const auto &x) { return false; },
                      },
                      box);
  }

  /// Does the boxed value have a rank greater than zero?
  bool hasRank() const {
    return std::visit(common::visitors{
                          [](const Intrinsic &) { return false; },
                          [](const Char &) { return false; },
                          [](const None &) { return false; },
                          [](const auto &x) { return x.extents.size() > 0; },
                      },
                      box);
  }

  /// Does the boxed value have trivial lower bounds (== 1)?
  bool hasSimpleLBounds() const {
    if (auto *arr = std::get_if<FullDim>(&box))
      return arr->lbounds.empty();
    if (auto *arr = std::get_if<CharFullDim>(&box))
      return arr->lbounds.empty();
    if (auto *arr = std::get_if<Derived>(&box))
      return (arr->extents.size() > 0) && arr->lbounds.empty();
    return false;
  }

  /// Does the boxed value have a constant shape?
  bool hasConstantShape() const {
    if (auto eleTy = fir::dyn_cast_ptrEleTy(getAddr().getType()))
      if (auto arrTy = eleTy.dyn_cast<fir::SequenceType>())
        return arrTy.hasConstantShape();
    return false;
  }

  /// Get the lbound if the box explicitly contains it.
  mlir::Value getLBound(unsigned dim) const {
    return std::visit(
        common::visitors{
            [&](const FullDim &box) { return box.lbounds[dim]; },
            [&](const CharFullDim &box) { return box.lbounds[dim]; },
            [&](const Derived &box) { return box.lbounds[dim]; },
            [](const auto &) { return mlir::Value{}; }},
        box);
  }

  /// Apply the lambda `func` to this box value.
  template <typename ON, typename RT>
  constexpr RT apply(RT(&&func)(const ON &)) const {
    if (auto *x = std::get_if<ON>(&box))
      return func(*x);
    return RT{};
  }

  std::variant<Intrinsic, FullDim, Char, CharFullDim, Derived, None> box;
};

//===----------------------------------------------------------------------===//
// Map of symbol information
//===----------------------------------------------------------------------===//

/// Helper class to map front-end symbols to their MLIR representation. This
/// provides a way to lookup the ssa-values that comprise a Fortran symbol's
/// runtime attributes. These attributes include its address, its dynamic size,
/// dynamic bounds information for non-scalar entities, dynamic type parameters,
/// etc.
class SymMap {
public:
  /// Add a trivial symbol mapping to an address.
  void addSymbol(semantics::SymbolRef sym, mlir::Value value,
                 bool force = false) {
    makeSym(sym, SymbolBox::Intrinsic(value), force);
  }

  /// Add a scalar CHARACTER mapping to an (address, len).
  void addCharSymbol(semantics::SymbolRef sym, mlir::Value value,
                     mlir::Value len, bool force = false) {
    makeSym(sym, SymbolBox::Char(value, len), force);
  }

  /// Add an array mapping with (address, shape).
  void addSymbolWithShape(semantics::SymbolRef sym, mlir::Value value,
                          llvm::ArrayRef<mlir::Value> shape,
                          bool force = false) {
    makeSym(sym, SymbolBox::FullDim(value, shape), force);
  }

  /// Add an array of CHARACTER mapping.
  void addCharSymbolWithShape(semantics::SymbolRef sym, mlir::Value value,
                              mlir::Value len,
                              llvm::ArrayRef<mlir::Value> shape,
                              bool force = false) {
    makeSym(sym, SymbolBox::CharFullDim(value, len, shape), force);
  }

  /// Add an array mapping with bounds notation.
  void addSymbolWithBounds(semantics::SymbolRef sym, mlir::Value value,
                           llvm::ArrayRef<mlir::Value> extents,
                           llvm::ArrayRef<mlir::Value> lbounds,
                           bool force = false) {
    makeSym(sym, SymbolBox::FullDim(value, extents, lbounds), force);
  }

  /// Add an array of CHARACTER with bounds notation.
  void addCharSymbolWithBounds(semantics::SymbolRef sym, mlir::Value value,
                               mlir::Value len,
                               llvm::ArrayRef<mlir::Value> extents,
                               llvm::ArrayRef<mlir::Value> lbounds,
                               bool force = false) {
    makeSym(sym, SymbolBox::CharFullDim(value, len, extents, lbounds), force);
  }

  /// Generalized derived type mapping.
  void addDerivedSymbol(semantics::SymbolRef sym, mlir::Value value,
                        mlir::Value size, llvm::ArrayRef<mlir::Value> extents,
                        llvm::ArrayRef<mlir::Value> lbounds,
                        llvm::ArrayRef<mlir::Value> params,
                        bool force = false) {
    makeSym(sym, SymbolBox::Derived(value, size, params, extents, lbounds),
            force);
  }

  /// Find `symbol` and return its value if it appears in the current mappings.
  SymbolBox lookupSymbol(semantics::SymbolRef sym) {
    auto iter = symbolMap.find(&*sym);
    return (iter == symbolMap.end()) ? SymbolBox() : iter->second;
  }

  /// Remove `sym` from the map.
  void erase(semantics::SymbolRef sym) { symbolMap.erase(&*sym); }

  /// Remove all symbols from the map.
  void clear() { symbolMap.clear(); }

  /// Dump the map. For debugging.
  void dump() const;

private:
  /// Add `symbol` to the current map and bind a `box`.
  void makeSym(semantics::SymbolRef sym, const SymbolBox &box,
               bool force = false) {
    if (force)
      erase(sym);
    assert(box && "cannot add an undefined symbol box");
    symbolMap.try_emplace(&*sym, box);
  }

  llvm::DenseMap<const semantics::Symbol *, SymbolBox> symbolMap;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_SYMBOLMAP_H
