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
    Reference,
    Pointer,
    Struct,

    // Synthetic boolean values are either atomic values or logical connectives.
    TopBool,
    AtomicBool,
    Conjunction,
    Disjunction,
    Negation,
    Implication,
    Biconditional,
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
    auto It = Properties.find(Name);
    return It == Properties.end() ? nullptr : It->second;
  }

  /// Assigns `Val` as the value of the synthetic property with the given
  /// `Name`.
  void setProperty(llvm::StringRef Name, Value &Val) {
    Properties.insert_or_assign(Name, &Val);
  }

private:
  Kind ValKind;
  llvm::StringMap<Value *> Properties;
};

/// Models a boolean.
class BoolValue : public Value {
public:
  explicit BoolValue(Kind ValueKind) : Value(ValueKind) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::TopBool ||
           Val->getKind() == Kind::AtomicBool ||
           Val->getKind() == Kind::Conjunction ||
           Val->getKind() == Kind::Disjunction ||
           Val->getKind() == Kind::Negation ||
           Val->getKind() == Kind::Implication ||
           Val->getKind() == Kind::Biconditional;
  }
};

/// Models the trivially true formula, which is Top in the lattice of boolean
/// formulas.
///
/// FIXME: Given the subtlety of comparison involving `TopBoolValue`, define
/// `operator==` for `Value`.
class TopBoolValue final : public BoolValue {
public:
  TopBoolValue() : BoolValue(Kind::TopBool) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::TopBool;
  }
};

/// Models an atomic boolean.
class AtomicBoolValue : public BoolValue {
public:
  explicit AtomicBoolValue() : BoolValue(Kind::AtomicBool) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::AtomicBool;
  }
};

/// Models a boolean conjunction.
// FIXME: Consider representing binary and unary boolean operations similar
// to how they are represented in the AST. This might become more pressing
// when such operations need to be added for other data types.
class ConjunctionValue : public BoolValue {
public:
  explicit ConjunctionValue(BoolValue &LeftSubVal, BoolValue &RightSubVal)
      : BoolValue(Kind::Conjunction), LeftSubVal(LeftSubVal),
        RightSubVal(RightSubVal) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Conjunction;
  }

  /// Returns the left sub-value of the conjunction.
  BoolValue &getLeftSubValue() const { return LeftSubVal; }

  /// Returns the right sub-value of the conjunction.
  BoolValue &getRightSubValue() const { return RightSubVal; }

private:
  BoolValue &LeftSubVal;
  BoolValue &RightSubVal;
};

/// Models a boolean disjunction.
class DisjunctionValue : public BoolValue {
public:
  explicit DisjunctionValue(BoolValue &LeftSubVal, BoolValue &RightSubVal)
      : BoolValue(Kind::Disjunction), LeftSubVal(LeftSubVal),
        RightSubVal(RightSubVal) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Disjunction;
  }

  /// Returns the left sub-value of the disjunction.
  BoolValue &getLeftSubValue() const { return LeftSubVal; }

  /// Returns the right sub-value of the disjunction.
  BoolValue &getRightSubValue() const { return RightSubVal; }

private:
  BoolValue &LeftSubVal;
  BoolValue &RightSubVal;
};

/// Models a boolean negation.
class NegationValue : public BoolValue {
public:
  explicit NegationValue(BoolValue &SubVal)
      : BoolValue(Kind::Negation), SubVal(SubVal) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Negation;
  }

  /// Returns the sub-value of the negation.
  BoolValue &getSubVal() const { return SubVal; }

private:
  BoolValue &SubVal;
};

/// Models a boolean implication.
///
/// Equivalent to `!LHS v RHS`.
class ImplicationValue : public BoolValue {
public:
  explicit ImplicationValue(BoolValue &LeftSubVal, BoolValue &RightSubVal)
      : BoolValue(Kind::Implication), LeftSubVal(LeftSubVal),
        RightSubVal(RightSubVal) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Implication;
  }

  /// Returns the left sub-value of the implication.
  BoolValue &getLeftSubValue() const { return LeftSubVal; }

  /// Returns the right sub-value of the implication.
  BoolValue &getRightSubValue() const { return RightSubVal; }

private:
  BoolValue &LeftSubVal;
  BoolValue &RightSubVal;
};

/// Models a boolean biconditional.
///
/// Equivalent to `(LHS ^ RHS) v (!LHS ^ !RHS)`.
class BiconditionalValue : public BoolValue {
public:
  explicit BiconditionalValue(BoolValue &LeftSubVal, BoolValue &RightSubVal)
      : BoolValue(Kind::Biconditional), LeftSubVal(LeftSubVal),
        RightSubVal(RightSubVal) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Biconditional;
  }

  /// Returns the left sub-value of the biconditional.
  BoolValue &getLeftSubValue() const { return LeftSubVal; }

  /// Returns the right sub-value of the biconditional.
  BoolValue &getRightSubValue() const { return RightSubVal; }

private:
  BoolValue &LeftSubVal;
  BoolValue &RightSubVal;
};

/// Models an integer.
class IntegerValue : public Value {
public:
  explicit IntegerValue() : Value(Kind::Integer) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Integer;
  }
};

/// Models a dereferenced pointer. For example, a reference in C++ or an lvalue
/// in C.
class ReferenceValue final : public Value {
public:
  explicit ReferenceValue(StorageLocation &ReferentLoc)
      : Value(Kind::Reference), ReferentLoc(ReferentLoc) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Reference;
  }

  StorageLocation &getReferentLoc() const { return ReferentLoc; }

private:
  StorageLocation &ReferentLoc;
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

/// Models a value of `struct` or `class` type, with a flat map of fields to
/// child storage locations, containing all accessible members of base struct
/// and class types.
class StructValue final : public Value {
public:
  StructValue() : StructValue(llvm::DenseMap<const ValueDecl *, Value *>()) {}

  explicit StructValue(llvm::DenseMap<const ValueDecl *, Value *> Children)
      : Value(Kind::Struct), Children(std::move(Children)) {}

  static bool classof(const Value *Val) {
    return Val->getKind() == Kind::Struct;
  }

  /// Returns the child value that is assigned for `D` or null if the child is
  /// not initialized.
  Value *getChild(const ValueDecl &D) const {
    auto It = Children.find(&D);
    if (It == Children.end())
      return nullptr;
    return It->second;
  }

  /// Assigns `Val` as the child value for `D`.
  void setChild(const ValueDecl &D, Value &Val) { Children[&D] = &Val; }

private:
  llvm::DenseMap<const ValueDecl *, Value *> Children;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_VALUE_H
