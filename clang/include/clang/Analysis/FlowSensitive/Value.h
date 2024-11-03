//===-- Value.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for values computed by abstract interpretation
// during dataflow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H

#include "clang/AST/Decl.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <utility>

namespace clang {
namespace dataflow {

/// Base class for all values computed by abstract interpretation.
///
/// Don't use `Value` instances by value. All `Value` instances are allocated
/// and owned by `DataflowAnalysisContext`.
class Value {
public:
  enum class Kind {
    Integer,
    Pointer,
    Record,

    // TODO: Top values should not be need to be type-specific.
    TopBool,
    AtomicBool,
    FormulaBool,
  };

  explicit Value(Kind ValKind) : ValKind(ValKind) {}

  // Non-copyable because addresses of values are used as their identities
  // throughout framework and user code. The framework is responsible for
  // construction and destruction of values.
  Value(const Value &) = delete;
  Value &operator=(const Value &) = delete;

  virtual ~Value() = default;

  Kind getKind() const { return ValKind; }

  /// Returns the value of the synthetic property with the given `Name` or null
  /// if the property isn't assigned a value.
  Value *getProperty(llvm::StringRef Name) const {
    return Properties.lookup(Name);
  }

  /// Assigns `Val` as the value of the synthetic property with the given
  /// `Name`.
  void setProperty(llvm::StringRef Name, Value &Val) {
    Properties.insert_or_assign(Name, &Val);
  }

  llvm::iterator_range<llvm::StringMap<Value *>::const_iterator>
  properties() const {
    return {Properties.begin(), Properties.end()};
  }

private:
  Kind ValKind;
  llvm::StringMap<Value *> Properties;
};

/// An equivalence relation for values. It obeys reflexivity, symmetry and
/// transitivity. It does *not* include comparison of `Properties`.
///
/// Computes equivalence for these subclasses:
/// * PointerValue -- pointee locations are equal. Does not compute deep
///   equality of `Value` at said location.
/// * TopBoolValue -- both are `TopBoolValue`s.
///
/// Otherwise, falls back to pointer equality.
bool areEquivalentValues(const Value &Val1, const Value &Val2);

/// Models a boolean.
class BoolValue : public Value {
  const Formula *F;

public:
  explicit BoolValue(Kind ValueKind, const Formula &F)
      : Value(ValueKind), F(&F) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::TopBool ||
           Val->getKind() == Kind::AtomicBool ||
           Val->getKind() == Kind::FormulaBool;
  }

  const Formula &formula() const { return *F; }
};

/// A TopBoolValue represents a boolean that is explicitly unconstrained.
///
/// This is equivalent to an AtomicBoolValue that does not appear anywhere
/// else in a system of formula.
/// Knowing the value is unconstrained is useful when e.g. reasoning about
/// convergence.
class TopBoolValue final : public BoolValue {
public:
  TopBoolValue(const Formula &F) : BoolValue(Kind::TopBool, F) {
    assert(F.kind() == Formula::AtomRef);
  }

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::TopBool;
  }

  Atom getAtom() const { return formula().getAtom(); }
};

/// Models an atomic boolean.
///
/// FIXME: Merge this class into FormulaBoolValue.
///        When we want to specify atom identity, use Atom.
class AtomicBoolValue final : public BoolValue {
public:
  explicit AtomicBoolValue(const Formula &F) : BoolValue(Kind::AtomicBool, F) {
    assert(F.kind() == Formula::AtomRef);
  }

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::AtomicBool;
  }

  Atom getAtom() const { return formula().getAtom(); }
};

/// Models a compound boolean formula.
class FormulaBoolValue final : public BoolValue {
public:
  explicit FormulaBoolValue(const Formula &F)
      : BoolValue(Kind::FormulaBool, F) {
    assert(F.kind() != Formula::AtomRef && "For now, use AtomicBoolValue");
  }

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::FormulaBool;
  }
};

/// Models an integer.
class IntegerValue : public Value {
public:
  explicit IntegerValue() : Value(Kind::Integer) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Integer;
  }
};

/// Models a symbolic pointer. Specifically, any value of type `T*`.
class PointerValue final : public Value {
public:
  explicit PointerValue(StorageLocation &PointeeLoc)
      : Value(Kind::Pointer), PointeeLoc(PointeeLoc) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Pointer;
  }

  StorageLocation &getPointeeLoc() const { return PointeeLoc; }

private:
  StorageLocation &PointeeLoc;
};

/// Models a value of `struct` or `class` type.
/// In C++, prvalues of class type serve only a limited purpose: They can only
/// be used to initialize a result object. It is not possible to access member
/// variables or call member functions on a prvalue of class type.
/// Correspondingly, `RecordValue` also serves only two limited purposes:
/// - It conveys a prvalue of class type from the place where the object is
///   constructed to the result object that it initializes.
///
///   When creating a prvalue of class type, we already need a storage location
///   for `this`, even though prvalues are otherwise not associated with storage
///   locations. `RecordValue` is therefore essentially a wrapper for a storage
///   location, which is then used to set the storage location for the result
///   object when we process the AST node for that result object.
///
///   For example:
///      MyStruct S = MyStruct(3);
///
///   In this example, `MyStruct(3) is a prvalue, which is modeled as a
///   `RecordValue` that wraps a `RecordStorageLocation`. This
//    `RecordStorageLocation` is then used as the storage location for `S`.
///
/// - It allows properties to be associated with an object of class type.
///   Note that when doing so, you should avoid mutating the properties of an
///   existing `RecordValue` in place, as these changes would be visible to
///   other `Environment`s that share the same `RecordValue`. Instead, associate
///   a new `RecordValue` with the `RecordStorageLocation` and set the
///   properties on this new `RecordValue`. (See also `refreshRecordValue()` in
///   DataflowEnvironment.h, which makes this easy.)
///   Note also that this implies that it is common for the same
///   `RecordStorageLocation` to be associated with different `RecordValue`s
///   in different environments.
/// Over time, we may eliminate `RecordValue` entirely. See also the discussion
/// here: https://reviews.llvm.org/D155204#inline-1503204
class RecordValue final : public Value {
public:
  explicit RecordValue(RecordStorageLocation &Loc)
      : Value(Kind::Record), Loc(Loc) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Record;
  }

  /// Returns the storage location that this `RecordValue` is associated with.
  RecordStorageLocation &getLoc() const { return Loc; }

  /// Convenience function that returns the child storage location for `Field`.
  /// See also the documentation for `RecordStorageLocation::getChild()`.
  StorageLocation *getChild(const ValueDecl &Field) const {
    return Loc.getChild(Field);
  }

private:
  RecordStorageLocation &Loc;
};

raw_ostream &operator<<(raw_ostream &OS, const Value &Val);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H
