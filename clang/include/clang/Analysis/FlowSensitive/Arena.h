//===-- Arena.h -------------------------------*- C++ -------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include <vector>

namespace clang::dataflow {

/// The Arena owns the objects that model data within an analysis.
/// For example, `Value` and `StorageLocation`.
class Arena {
public:
  Arena()
      : TrueVal(create<AtomicBoolValue>()),
        FalseVal(create<AtomicBoolValue>()) {}
  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;

  /// Creates a `T` (some subclass of `StorageLocation`), forwarding `args` to
  /// the constructor, and returns a reference to it.
  ///
  /// The `DataflowAnalysisContext` takes ownership of the created object. The
  /// object will be destroyed when the `DataflowAnalysisContext` is destroyed.
  template <typename T, typename... Args>
  std::enable_if_t<std::is_base_of<StorageLocation, T>::value, T &>
  create(Args &&...args) {
    // Note: If allocation of individual `StorageLocation`s turns out to be
    // costly, consider creating specializations of `create<T>` for commonly
    // used `StorageLocation` subclasses and make them use a `BumpPtrAllocator`.
    return *cast<T>(
        Locs.emplace_back(std::make_unique<T>(std::forward<Args>(args)...))
            .get());
  }

  /// Creates a `T` (some subclass of `Value`), forwarding `args` to the
  /// constructor, and returns a reference to it.
  ///
  /// The `DataflowAnalysisContext` takes ownership of the created object. The
  /// object will be destroyed when the `DataflowAnalysisContext` is destroyed.
  template <typename T, typename... Args>
  std::enable_if_t<std::is_base_of<Value, T>::value, T &>
  create(Args &&...args) {
    // Note: If allocation of individual `Value`s turns out to be costly,
    // consider creating specializations of `create<T>` for commonly used
    // `Value` subclasses and make them use a `BumpPtrAllocator`.
    return *cast<T>(
        Vals.emplace_back(std::make_unique<T>(std::forward<Args>(args)...))
            .get());
  }

  /// Returns a boolean value that represents the conjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &makeAnd(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents the disjunction of `LHS` and
  /// `RHS`. Subsequent calls with the same arguments, regardless of their
  /// order, will return the same result. If the given boolean values represent
  /// the same value, the result will be the value itself.
  BoolValue &makeOr(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents the negation of `Val`. Subsequent
  /// calls with the same argument will return the same result.
  BoolValue &makeNot(BoolValue &Val);

  /// Returns a boolean value that represents `LHS => RHS`. Subsequent calls
  /// with the same arguments, will return the same result. If the given boolean
  /// values represent the same value, the result will be a value that
  /// represents the true boolean literal.
  BoolValue &makeImplies(BoolValue &LHS, BoolValue &RHS);

  /// Returns a boolean value that represents `LHS <=> RHS`. Subsequent calls
  /// with the same arguments, regardless of their order, will return the same
  /// result. If the given boolean values represent the same value, the result
  /// will be a value that represents the true boolean literal.
  BoolValue &makeEquals(BoolValue &LHS, BoolValue &RHS);

  /// Returns a symbolic boolean value that models a boolean literal equal to
  /// `Value`. These literals are the same every time.
  AtomicBoolValue &makeLiteral(bool Value) const {
    return Value ? TrueVal : FalseVal;
  }

  /// Creates a fresh flow condition and returns a token that identifies it. The
  /// token can be used to perform various operations on the flow condition such
  /// as adding constraints to it, forking it, joining it with another flow
  /// condition, or checking implications.
  AtomicBoolValue &makeFlowConditionToken() {
    return create<AtomicBoolValue>();
  }

private:
  // Storage for the state of a program.
  std::vector<std::unique_ptr<StorageLocation>> Locs;
  std::vector<std::unique_ptr<Value>> Vals;

  // Indices that are used to avoid recreating the same composite boolean
  // values.
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, ConjunctionValue *>
      ConjunctionVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, DisjunctionValue *>
      DisjunctionVals;
  llvm::DenseMap<BoolValue *, NegationValue *> NegationVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, ImplicationValue *>
      ImplicationVals;
  llvm::DenseMap<std::pair<BoolValue *, BoolValue *>, BiconditionalValue *>
      BiconditionalVals;

  AtomicBoolValue &TrueVal;
  AtomicBoolValue &FalseVal;
};

} // namespace clang::dataflow
