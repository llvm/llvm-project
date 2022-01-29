//===-- CIRGenValue.h - CIRGen something TODO this desc* --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IDK yet
// TODO:
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENVALUE_H
#define LLVM_CLANG_LIB_CIR_CIRGENVALUE_H

#include "Address.h"
#include "CIRGenFunction.h"

#include "mlir/IR/Value.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/PointerIntPair.h"

namespace cir {

/// This trivial value class is used to represent the result of an
/// expression that is evaluated.  It can be one of three things: either a
/// simple MLIR SSA value, a pair of SSA values for complex numbers, or the
/// address of an aggregate value in memory.
class RValue {
  enum Flavor { Scalar, Complex, Aggregate };

  // The shift to make to an aggregate's alignment to make it look
  // like a pointer.
  enum { AggAlignShift = 4 };

  // Stores first value and flavor.
  llvm::PointerIntPair<mlir::Value, 2, Flavor> V1;
  // Stores second value and volatility.
  llvm::PointerIntPair<mlir::Value, 1, bool> V2;

public:
  bool isScalar() const { return V1.getInt() == Scalar; }
  bool isComplex() const { return V1.getInt() == Complex; }
  bool isAggregate() const { return V1.getInt() == Aggregate; }

  bool isVolatileQualified() const { return V2.getInt(); }

  /// getScalarVal() - Return the Value* of this scalar value.
  mlir::Value getScalarVal() const {
    assert(isScalar() && "Not a scalar!");
    return V1.getPointer();
  }

  /// getComplexVal - Return the real/imag components of this complex value.
  ///
  std::pair<mlir::Value, mlir::Value> getComplexVal() const {
    assert(0 && "not implemented");
    return {};
  }

  /// getAggregateAddr() - Return the Value* of the address of the
  /// aggregate.
  Address getAggregateAddress() const {
    assert(0 && "not implemented");
    return Address::invalid();
  }

  static RValue getIgnored() {
    // FIXME: should we make this a more explicit state?
    return get(nullptr);
  }

  static RValue get(mlir::Value V) {
    RValue ER;
    ER.V1.setPointer(V);
    ER.V1.setInt(Scalar);
    ER.V2.setInt(false);
    return ER;
  }
  static RValue getComplex(mlir::Value V1, mlir::Value V2) {
    assert(0 && "not implemented");
    return RValue{};
  }
  static RValue getComplex(const std::pair<mlir::Value, mlir::Value> &C) {
    assert(0 && "not implemented");
    return RValue{};
  }
  // FIXME: Aggregate rvalues need to retain information about whether they
  // are volatile or not.  Remove default to find all places that probably
  // get this wrong.
  static RValue getAggregate(Address addr, bool isVolatile = false) {
    assert(0 && "not implemented");
    return RValue{};
  }
};

/// The source of the alignment of an l-value; an expression of
/// confidence in the alignment actually matching the estimate.
enum class AlignmentSource {
  /// The l-value was an access to a declared entity or something
  /// equivalently strong, like the address of an array allocated by a
  /// language runtime.
  Decl,

  /// The l-value was considered opaque, so the alignment was
  /// determined from a type, but that type was an explicitly-aligned
  /// typedef.
  AttributedType,

  /// The l-value was considered opaque, so the alignment was
  /// determined from a type.
  Type
};

/// Given that the base address has the given alignment source, what's
/// our confidence in the alignment of the field?
static inline AlignmentSource getFieldAlignmentSource(AlignmentSource Source) {
  // For now, we don't distinguish fields of opaque pointers from
  // top-level declarations, but maybe we should.
  return AlignmentSource::Decl;
}

class LValueBaseInfo {
  AlignmentSource AlignSource;

public:
  explicit LValueBaseInfo(AlignmentSource Source = AlignmentSource::Type)
      : AlignSource(Source) {}
  AlignmentSource getAlignmentSource() const { return AlignSource; }
  void setAlignmentSource(AlignmentSource Source) { AlignSource = Source; }

  void mergeForCast(const LValueBaseInfo &Info) {
    setAlignmentSource(Info.getAlignmentSource());
  }
};

class LValue {
  enum {
    Simple,       // This is a normal l-value, use getAddress().
    VectorElt,    // This is a vector element l-value (V[i]), use getVector*
    BitField,     // This is a bitfield l-value, use getBitfield*.
    ExtVectorElt, // This is an extended vector subset, use getExtVectorComp
    GlobalReg,    // This is a register l-value, use getGlobalReg()
    MatrixElt     // This is a matrix element, use getVector*
  } LVType;
  clang::QualType Type;
  clang::Qualifiers Quals;

private:
  void Initialize(clang::CharUnits Alignment, clang::QualType Type,
                  LValueBaseInfo BaseInfo) {
    // assert((!Alignment.isZero()) && // || Type->isIncompleteType()) &&
    //       "initializing l-value with zero alignment!");
    this->Type = Type;
    // This flag shows if a nontemporal load/stores should be used when
    // accessing this lvalue.
    const unsigned MaxAlign = 1U << 31;
    this->Alignment = Alignment.getQuantity() <= MaxAlign
                          ? Alignment.getQuantity()
                          : MaxAlign;
    assert(this->Alignment == Alignment.getQuantity() &&
           "Alignment exceeds allowed max!");
    this->BaseInfo = BaseInfo;
  }

  // The alignment to use when accessing this lvalue. (For vector elements,
  // this is the alignment of the whole vector)
  unsigned Alignment;
  mlir::Value V;
  LValueBaseInfo BaseInfo;

public:
  bool isSimple() const { return LVType == Simple; }
  bool isVectorElt() const { return LVType == VectorElt; }
  bool isBitField() const { return LVType == BitField; }
  bool isExtVectorElt() const { return LVType == ExtVectorElt; }
  bool isGlobalReg() const { return LVType == GlobalReg; }
  bool isMatrixElt() const { return LVType == MatrixElt; }

  clang::QualType getType() const { return Type; }

  mlir::Value getPointer() const { return V; }

  clang::CharUnits getAlignment() const {
    return clang::CharUnits::fromQuantity(Alignment);
  }

  Address getAddress() const { return Address(getPointer(), getAlignment()); }

  LValueBaseInfo getBaseInfo() const { return BaseInfo; }
  void setBaseInfo(LValueBaseInfo Info) { BaseInfo = Info; }

  static LValue makeAddr(Address address, clang::QualType T,
                         AlignmentSource Source = AlignmentSource::Type) {
    LValue R;
    R.V = address.getPointer();
    R.Initialize(address.getAlignment(), T, LValueBaseInfo(Source));
    R.LVType = Simple;
    return R;
  }

  // FIXME: only have one of these static methods.
  static LValue makeAddr(Address address, clang::QualType T,
                         LValueBaseInfo LBI) {
    LValue R;
    R.V = address.getPointer();
    R.Initialize(address.getAlignment(), T, LBI);
    R.LVType = Simple;
    return R;
  }

  const clang::Qualifiers &getQuals() const { return Quals; }
  clang::Qualifiers &getQuals() { return Quals; }
};

} // namespace cir

#endif
