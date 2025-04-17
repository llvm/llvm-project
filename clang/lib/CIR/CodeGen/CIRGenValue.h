//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes implement wrappers around mlir::Value in order to fully
// represent the range of values for C L- and R- values.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CIRGENVALUE_H
#define CLANG_LIB_CIR_CIRGENVALUE_H

#include "Address.h"

#include "clang/AST/CharUnits.h"
#include "clang/AST/Type.h"

#include "llvm/ADT/PointerIntPair.h"

#include "mlir/IR/Value.h"

#include "clang/CIR/MissingFeatures.h"

namespace clang::CIRGen {

/// This trivial value class is used to represent the result of an
/// expression that is evaluated. It can be one of three things: either a
/// simple MLIR SSA value, a pair of SSA values for complex numbers, or the
/// address of an aggregate value in memory.
class RValue {
  enum Flavor { Scalar, Complex, Aggregate };

  // Stores first value and flavor.
  llvm::PointerIntPair<mlir::Value, 2, Flavor> v1;
  // Stores second value and volatility.
  llvm::PointerIntPair<llvm::PointerUnion<mlir::Value, int *>, 1, bool> v2;
  // Stores element type for aggregate values.
  mlir::Type elementType;

public:
  bool isScalar() const { return v1.getInt() == Scalar; }

  /// Return the mlir::Value of this scalar value.
  mlir::Value getScalarVal() const {
    assert(isScalar() && "Not a scalar!");
    return v1.getPointer();
  }

  static RValue get(mlir::Value v) {
    RValue er;
    er.v1.setPointer(v);
    er.v1.setInt(Scalar);
    er.v2.setInt(false);
    return er;
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
static inline AlignmentSource getFieldAlignmentSource(AlignmentSource source) {
  // For now, we don't distinguish fields of opaque pointers from
  // top-level declarations, but maybe we should.
  return AlignmentSource::Decl;
}

class LValueBaseInfo {
  AlignmentSource alignSource;

public:
  explicit LValueBaseInfo(AlignmentSource source = AlignmentSource::Type)
      : alignSource(source) {}
  AlignmentSource getAlignmentSource() const { return alignSource; }
  void setAlignmentSource(AlignmentSource source) { alignSource = source; }

  void mergeForCast(const LValueBaseInfo &info) {
    setAlignmentSource(info.getAlignmentSource());
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
  } lvType;
  clang::QualType type;
  clang::Qualifiers quals;

  // The alignment to use when accessing this lvalue. (For vector elements,
  // this is the alignment of the whole vector)
  unsigned alignment;
  mlir::Value v;
  mlir::Type elementType;
  LValueBaseInfo baseInfo;

  void initialize(clang::QualType type, clang::Qualifiers quals,
                  clang::CharUnits alignment, LValueBaseInfo baseInfo) {
    assert((!alignment.isZero() || type->isIncompleteType()) &&
           "initializing l-value with zero alignment!");
    this->type = type;
    this->quals = quals;
    const unsigned maxAlign = 1U << 31;
    this->alignment = alignment.getQuantity() <= maxAlign
                          ? alignment.getQuantity()
                          : maxAlign;
    assert(this->alignment == alignment.getQuantity() &&
           "Alignment exceeds allowed max!");
    this->baseInfo = baseInfo;
  }

public:
  bool isSimple() const { return lvType == Simple; }
  bool isBitField() const { return lvType == BitField; }

  // TODO: Add support for volatile
  bool isVolatile() const { return false; }

  clang::QualType getType() const { return type; }

  mlir::Value getPointer() const { return v; }

  clang::CharUnits getAlignment() const {
    return clang::CharUnits::fromQuantity(alignment);
  }
  void setAlignment(clang::CharUnits a) { alignment = a.getQuantity(); }

  Address getAddress() const {
    return Address(getPointer(), elementType, getAlignment());
  }

  const clang::Qualifiers &getQuals() const { return quals; }

  static LValue makeAddr(Address address, clang::QualType t,
                         LValueBaseInfo baseInfo) {
    // Classic codegen sets the objc gc qualifier here. That requires an
    // ASTContext, which is passed in from CIRGenFunction::makeAddrLValue.
    assert(!cir::MissingFeatures::objCGC());

    LValue r;
    r.lvType = Simple;
    r.v = address.getPointer();
    r.elementType = address.getElementType();
    r.initialize(t, t.getQualifiers(), address.getAlignment(), baseInfo);
    return r;
  }
};

/// An aggregate value slot.
class AggValueSlot {

  Address addr;
  clang::Qualifiers quals;

  /// This is set to true if the memory in the slot is known to be zero before
  /// the assignment into it.  This means that zero fields don't need to be set.
  bool zeroedFlag : 1;

public:
  enum IsZeroed_t { IsNotZeroed, IsZeroed };

  AggValueSlot(Address addr, clang::Qualifiers quals, bool zeroedFlag)
      : addr(addr), quals(quals), zeroedFlag(zeroedFlag) {}

  static AggValueSlot forAddr(Address addr, clang::Qualifiers quals,
                              IsZeroed_t isZeroed = IsNotZeroed) {
    return AggValueSlot(addr, quals, isZeroed);
  }

  static AggValueSlot forLValue(const LValue &lv) {
    return forAddr(lv.getAddress(), lv.getQuals());
  }

  clang::Qualifiers getQualifiers() const { return quals; }

  Address getAddress() const { return addr; }

  bool isIgnored() const { return !addr.isValid(); }

  IsZeroed_t isZeroed() const { return IsZeroed_t(zeroedFlag); }
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CIRGENVALUE_H
