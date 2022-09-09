//===-- CIRGenValue.h - CIRGen wrappers for mlir::Value ---------*- C++ -*-===//
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

#ifndef LLVM_CLANG_LIB_CIR_CIRGENVALUE_H
#define LLVM_CLANG_LIB_CIR_CIRGENVALUE_H

#include "Address.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Type.h"

#include "llvm/ADT/PointerIntPair.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/IR/Value.h"

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

  // This flag shows if a nontemporal load/stores should be used when accessing
  // this lvalue.
  bool Nontemporal : 1;

private:
  void Initialize(clang::QualType Type, clang::Qualifiers Quals,
                  clang::CharUnits Alignment, LValueBaseInfo BaseInfo) {
    assert((!Alignment.isZero() || Type->isIncompleteType()) &&
           "initializing l-value with zero alignment!");
    if (isGlobalReg())
      assert(ElementType == nullptr && "Glboal reg does not store elem type");
    else
      assert(V.getType().cast<mlir::cir::PointerType>().getPointee() ==
                 ElementType &&
             "Pointer element type mismatch");

    this->Type = Type;
    this->Quals = Quals;
    // This flag shows if a nontemporal load/stores should be used when
    // accessing this lvalue.
    const unsigned MaxAlign = 1U << 31;
    this->Alignment = Alignment.getQuantity() <= MaxAlign
                          ? Alignment.getQuantity()
                          : MaxAlign;
    assert(this->Alignment == Alignment.getQuantity() &&
           "Alignment exceeds allowed max!");
    this->BaseInfo = BaseInfo;

    // TODO: ObjC flags
    // Initialize Objective-C flags.
    this->Nontemporal = false;
  }

  // The alignment to use when accessing this lvalue. (For vector elements,
  // this is the alignment of the whole vector)
  unsigned Alignment;
  mlir::Value V;
  mlir::Type ElementType;
  LValueBaseInfo BaseInfo;

public:
  bool isSimple() const { return LVType == Simple; }
  bool isVectorElt() const { return LVType == VectorElt; }
  bool isBitField() const { return LVType == BitField; }
  bool isExtVectorElt() const { return LVType == ExtVectorElt; }
  bool isGlobalReg() const { return LVType == GlobalReg; }
  bool isMatrixElt() const { return LVType == MatrixElt; }

  unsigned getVRQualifiers() const {
    return Quals.getCVRQualifiers() & ~clang::Qualifiers::Const;
  }

  bool isVolatile() const { return Quals.hasVolatile(); }

  bool isNontemporal() const { return Nontemporal; }

  clang::QualType getType() const { return Type; }

  mlir::Value getPointer() const { return V; }

  clang::CharUnits getAlignment() const {
    return clang::CharUnits::fromQuantity(Alignment);
  }

  Address getAddress() const {
    return Address(getPointer(), ElementType, getAlignment());
  }

  LValueBaseInfo getBaseInfo() const { return BaseInfo; }
  void setBaseInfo(LValueBaseInfo Info) { BaseInfo = Info; }

  static LValue makeAddr(Address address, clang::QualType T,
                         AlignmentSource Source = AlignmentSource::Type) {
    LValue R;
    R.LVType = Simple;
    R.V = address.getPointer();
    R.ElementType = address.getElementType();
    R.Initialize(T, T.getQualifiers(), address.getAlignment(),
                 LValueBaseInfo(Source));
    return R;
  }

  // FIXME: only have one of these static methods.
  static LValue makeAddr(Address address, clang::QualType T,
                         LValueBaseInfo LBI) {
    LValue R;
    R.LVType = Simple;
    R.V = address.getPointer();
    R.ElementType = address.getElementType();
    R.Initialize(T, T.getQualifiers(), address.getAlignment(), LBI);
    return R;
  }

  static LValue makeAddr(Address address, clang::QualType type,
                         clang::ASTContext &Context, LValueBaseInfo BaseInfo) {
    clang::Qualifiers qs = type.getQualifiers();
    qs.setObjCGCAttr(Context.getObjCGCAttrKind(type));

    LValue R;
    R.LVType = Simple;
    assert(address.getPointer().getType().cast<mlir::cir::PointerType>());
    R.V = address.getPointer();
    R.ElementType = address.getElementType();
    R.Initialize(type, qs, address.getAlignment(),
                 BaseInfo); // TODO: TBAAInfo);
    return R;
  }

  const clang::Qualifiers &getQuals() const { return Quals; }
  clang::Qualifiers &getQuals() { return Quals; }
};

/// An aggregate value slot.
class AggValueSlot {
  /// The address.
  Address Addr;

  // Qualifiers
  clang::Qualifiers Quals;

  /// ZeroedFlag - This is set to true if the memory in the slot is known to be
  /// zero before the assignment into it. This means that zero field don't need
  /// to be set.
  bool ZeroedFlag : 1;

  /// This is set to true if the tail padding of this slot might overlap another
  /// object that may have already been initialized (and whose value must be
  /// preserved by this initialization). If so, we may only store up to the
  /// dsize of the type. Otherwise we can widen stores to the size of the type.
  bool OverlapFlag : 1;

  /// DestructedFlags - This is set to true if some external code is responsible
  /// for setting up a destructor for the slot. Otherwise the code which
  /// constructs it shoudl push the appropriate cleanup.
  // bool DestructedFlag : 1;

  /// If is set to true, sanitizer checks are already generated for this address
  /// or not required. For instance, if this address represents an object
  /// created in 'new' expression, sanitizer checks for memory is made as a part
  /// of 'operator new' emission and object constructor should not generate
  /// them.
  bool SanitizerCheckedFlag : 1;

  // TODO: Add the rest of these things

  AggValueSlot(Address Addr, clang::Qualifiers Quals, bool DestructedFlag,
               bool ObjCGCFlag, bool ZeroedFlag, bool AliasedFlag,
               bool OverlapFlag, bool SanitizerCheckedFlag)
      : Addr(Addr), Quals(Quals)
  // ,DestructedFlag(DestructedFlag)
  // ,ObjCGCFlag(ObjCGCFlag)
  // ,ZeroedFlag(ZeroedFlag)
  // ,AliasedFlag(AliasedFlag)
  // ,OverlapFlag(OverlapFlag)
  // ,SanitizerCheckedFlag(SanitizerCheckedFlag)
  {}

public:
  enum IsAliased_t { IsNotAliased, IsAliased };
  enum IsDestructed_t { IsNotDestructed, IsDestructed };
  enum IsZeroed_t { IsNotZeroed, IsZeroed };
  enum Overlap_t { DoesNotOverlap, MayOverlap };
  enum NeedsGCBarriers_t { DoesNotNeedGCBarriers, NeedsGCBarriers };
  enum IsSanitizerChecked_t { IsNotSanitizerChecked, IsSanitizerChecked };

  /// ignored - Returns an aggregate value slot indicating that the aggregate
  /// value is being ignored.
  static AggValueSlot ignored() {
    return forAddr(Address::invalid(), clang::Qualifiers(), IsNotDestructed,
                   DoesNotNeedGCBarriers, IsNotAliased, DoesNotOverlap);
  }

  /// forAddr - Make a slot for an aggregate value.
  ///
  /// \param quals - The qualifiers that dictate how the slot should be
  ///   initialized. Only 'volatile' and the Objective-C lifetime qualifiers
  ///   matter.
  ///
  /// \param isDestructed - true if something else is responsible for calling
  ///   destructors on this object
  /// \param needsGC - true fi the slot is potentially located somewhere that
  ///   ObjC GC calls should be emitted for
  static AggValueSlot
  forAddr(Address addr, clang::Qualifiers quals, IsDestructed_t isDestructed,
          NeedsGCBarriers_t needsGC, IsAliased_t isAliased,
          Overlap_t mayOverlap, IsZeroed_t isZeroed = IsNotZeroed,
          IsSanitizerChecked_t isChecked = IsNotSanitizerChecked) {
    return AggValueSlot(addr, quals, isDestructed, needsGC, isZeroed, isAliased,
                        mayOverlap, isChecked);
  }

  static AggValueSlot
  forLValue(const LValue &LV, IsDestructed_t isDestructed,
            NeedsGCBarriers_t needsGC, IsAliased_t isAliased,
            Overlap_t mayOverlap, IsZeroed_t isZeroed = IsNotZeroed,
            IsSanitizerChecked_t isChecked = IsNotSanitizerChecked) {
    return forAddr(LV.getAddress(), LV.getQuals(), isDestructed, needsGC,
                   isAliased, mayOverlap, isZeroed, isChecked);
  }

  clang::Qualifiers getQualifiers() const { return Quals; }

  bool isVolatile() const { return Quals.hasVolatile(); }

  Address getAddress() const { return Addr; }

  bool isIgnored() const { return !Addr.isValid(); }

  Overlap_t mayOverlap() const { return Overlap_t(OverlapFlag); }

  bool isSanitizerChecked() const { return SanitizerCheckedFlag; }

  IsZeroed_t isZeroed() const { return IsZeroed_t(ZeroedFlag); }
};

} // namespace cir

#endif
