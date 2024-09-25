//===-- SymbolMap.h -- lowering internal symbol map -------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_SYMBOLMAP_H
#define FORTRAN_LOWER_SYMBOLMAP_H

#include "flang/Common/reference.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/Support/Matcher.h"
#include "flang/Semantics/symbol.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace Fortran::lower {

struct SymbolBox;
class SymMap;
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SymbolBox &symMap);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SymMap &symMap);

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
struct SymbolBox : public fir::details::matcher<SymbolBox> {
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

  // Pointer or allocatable variable
  using PointerOrAllocatable = fir::MutableBoxValue;

  // Non pointer/allocatable variable that must be tracked with
  // a fir.box (either because it is not contiguous, or assumed rank, or assumed
  // type, or polymorphic, or because the fir.box is describing an optional
  // value and cannot be read into one of the other category when lowering the
  // symbol).
  using Box = fir::BoxValue;

  using VT =
      std::variant<Intrinsic, FullDim, Char, CharFullDim, PointerOrAllocatable,
                   Box, fir::FortranVariableOpInterface, None>;

  //===--------------------------------------------------------------------===//
  // Constructors
  //===--------------------------------------------------------------------===//

  SymbolBox() : box{None{}} {}
  template <typename A>
  SymbolBox(const A &x) : box{x} {}

  explicit operator bool() const { return !std::holds_alternative<None>(box); }

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  /// Get address of the boxed value. For a scalar, this is the address of the
  /// scalar. For an array, this is the address of the first element in the
  /// array, etc.
  mlir::Value getAddr() const {
    return match([](const None &) { return mlir::Value{}; },
                 [](const fir::FortranVariableOpInterface &x) {
                   return fir::FortranVariableOpInterface(x).getBase();
                 },
                 [](const auto &x) { return x.getAddr(); });
  }

  std::optional<fir::FortranVariableOpInterface>
  getIfFortranVariableOpInterface() {
    return match(
        [](const fir::FortranVariableOpInterface &x)
            -> std::optional<fir::FortranVariableOpInterface> { return x; },
        [](const auto &x) -> std::optional<fir::FortranVariableOpInterface> {
          return std::nullopt;
        });
  }

  /// Apply the lambda `func` to this box value.
  template <typename ON, typename RT>
  constexpr RT apply(RT (&&func)(const ON &)) const {
    if (auto *x = std::get_if<ON>(&box))
      return func(*x);
    return RT{};
  }

  const VT &matchee() const { return box; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SymbolBox &symBox);

  /// Dump the map. For debugging.
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this << '\n'; }

private:
  VT box;
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
  using AcDoVar = llvm::StringRef;

  SymMap() { pushScope(); }
  SymMap(const SymMap &) = delete;

  void pushScope() { symbolMapStack.emplace_back(); }
  void popScope() {
    symbolMapStack.pop_back();
    assert(symbolMapStack.size() >= 1);
  }

  /// Add an extended value to the symbol table.
  void addSymbol(semantics::SymbolRef sym, const fir::ExtendedValue &ext,
                 bool force = false);

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
  void addCharSymbol(semantics::SymbolRef sym, const SymbolBox::Char &value,
                     bool force = false) {
    makeSym(sym, value, force);
  }

  /// Add an array mapping with (address, shape).
  void addSymbolWithShape(semantics::SymbolRef sym, mlir::Value value,
                          llvm::ArrayRef<mlir::Value> shape,
                          bool force = false) {
    makeSym(sym, SymbolBox::FullDim(value, shape), force);
  }
  void addSymbolWithShape(semantics::SymbolRef sym,
                          const SymbolBox::FullDim &value, bool force = false) {
    makeSym(sym, value, force);
  }

  /// Add an array of CHARACTER mapping.
  void addCharSymbolWithShape(semantics::SymbolRef sym, mlir::Value value,
                              mlir::Value len,
                              llvm::ArrayRef<mlir::Value> shape,
                              bool force = false) {
    makeSym(sym, SymbolBox::CharFullDim(value, len, shape), force);
  }
  void addCharSymbolWithShape(semantics::SymbolRef sym,
                              const SymbolBox::CharFullDim &value,
                              bool force = false) {
    makeSym(sym, value, force);
  }

  /// Add an array mapping with bounds notation.
  void addSymbolWithBounds(semantics::SymbolRef sym, mlir::Value value,
                           llvm::ArrayRef<mlir::Value> extents,
                           llvm::ArrayRef<mlir::Value> lbounds,
                           bool force = false) {
    makeSym(sym, SymbolBox::FullDim(value, extents, lbounds), force);
  }
  void addSymbolWithBounds(semantics::SymbolRef sym,
                           const SymbolBox::FullDim &value,
                           bool force = false) {
    makeSym(sym, value, force);
  }

  /// Add an array of CHARACTER with bounds notation.
  void addCharSymbolWithBounds(semantics::SymbolRef sym, mlir::Value value,
                               mlir::Value len,
                               llvm::ArrayRef<mlir::Value> extents,
                               llvm::ArrayRef<mlir::Value> lbounds,
                               bool force = false) {
    makeSym(sym, SymbolBox::CharFullDim(value, len, extents, lbounds), force);
  }
  void addCharSymbolWithBounds(semantics::SymbolRef sym,
                               const SymbolBox::CharFullDim &value,
                               bool force = false) {
    makeSym(sym, value, force);
  }

  void addAllocatableOrPointer(semantics::SymbolRef sym,
                               fir::MutableBoxValue box, bool force = false) {
    makeSym(sym, box, force);
  }

  void addBoxSymbol(semantics::SymbolRef sym, mlir::Value irBox,
                    llvm::ArrayRef<mlir::Value> lbounds,
                    llvm::ArrayRef<mlir::Value> explicitParams,
                    llvm::ArrayRef<mlir::Value> explicitExtents,
                    bool force = false) {
    makeSym(sym,
            SymbolBox::Box(irBox, lbounds, explicitParams, explicitExtents),
            force);
  }
  void addBoxSymbol(semantics::SymbolRef sym, const SymbolBox::Box &value,
                    bool force = false) {
    makeSym(sym, value, force);
  }

  /// Find `symbol` and return its value if it appears in the current mappings.
  SymbolBox lookupSymbol(semantics::SymbolRef sym);
  SymbolBox lookupSymbol(const semantics::Symbol *sym) {
    return lookupSymbol(*sym);
  }

  /// Find `symbol` and return its value if it appears in the inner-most level
  /// map.
  SymbolBox shallowLookupSymbol(semantics::SymbolRef sym);
  SymbolBox shallowLookupSymbol(const semantics::Symbol *sym) {
    return shallowLookupSymbol(*sym);
  }

  /// Find `symbol` and return its value if it appears in the one level up map
  /// such as for the host variable in host-association in OpenMP code.
  SymbolBox lookupOneLevelUpSymbol(semantics::SymbolRef sym);
  SymbolBox lookupOneLevelUpSymbol(const semantics::Symbol *sym) {
    return lookupOneLevelUpSymbol(*sym);
  }

  /// Add a new binding from the ac-do-variable `var` to `value`.
  void pushImpliedDoBinding(AcDoVar var, mlir::Value value) {
    impliedDoStack.emplace_back(var, value);
  }

  /// Pop the most recent implied do binding off the stack.
  void popImpliedDoBinding() {
    assert(!impliedDoStack.empty());
    impliedDoStack.pop_back();
  }

  /// Lookup the ac-do-variable and return the Value it is bound to.
  /// If the variable is not found, returns a null Value.
  mlir::Value lookupImpliedDo(AcDoVar var);

  /// Remove all symbols from the map.
  void clear() {
    symbolMapStack.clear();
    symbolMapStack.emplace_back();
    assert(symbolMapStack.size() == 1);
    impliedDoStack.clear();
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const SymMap &symMap);

  /// Dump the map. For debugging.
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this << '\n'; }

  void addVariableDefinition(semantics::SymbolRef symRef,
                             fir::FortranVariableOpInterface definingOp,
                             bool force = false) {
    makeSym(symRef, SymbolBox(definingOp), force);
  }

  void copySymbolBinding(semantics::SymbolRef src,
                         semantics::SymbolRef target) {
    auto symBox = lookupSymbol(src);
    assert(symBox && "source binding does not exists");
    makeSym(target, symBox, /*force=*/false);
  }

  std::optional<fir::FortranVariableOpInterface>
  lookupVariableDefinition(semantics::SymbolRef sym) {
    if (auto symBox = lookupSymbol(sym))
      return symBox.getIfFortranVariableOpInterface();
    return std::nullopt;
  }

private:
  /// Bind `box` to `symRef` in the symbol map.
  void makeSym(semantics::SymbolRef symRef, const SymbolBox &box,
               bool force = false) {
    auto *sym = symRef->HasLocalLocality() ? &*symRef : &symRef->GetUltimate();
    if (force)
      symbolMapStack.back().erase(sym);
    assert(box && "cannot add an undefined symbol box");
    symbolMapStack.back().try_emplace(sym, box);
  }

  llvm::SmallVector<llvm::DenseMap<const semantics::Symbol *, SymbolBox>>
      symbolMapStack;

  // Implied DO induction variables are not represented as Se::Symbol in
  // Ev::Expr. Keep the variable markers in their own stack.
  llvm::SmallVector<std::pair<AcDoVar, mlir::Value>> impliedDoStack;
};

/// RAII wrapper for SymMap.
class SymMapScope {
public:
  explicit SymMapScope(SymMap &map) : map(map) { map.pushScope(); }
  ~SymMapScope() { map.popScope(); }

private:
  SymMap &map;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_SYMBOLMAP_H
