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
#include "CIRGenRecordLayout.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/ADT/PointerIntPair.h"

#include "mlir/IR/Value.h"

namespace cir {

/// This trivial value class is used to represent the result of an
/// expression that is evaluated. It can be one of three things: either a
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
  llvm::PointerIntPair<llvm::PointerUnion<mlir::Value, int *>, 1, bool> V2;
  // Stores element type for aggregate values.
  mlir::Type ElementType;

public:
  bool isScalar() const { return V1.getInt() == Scalar; }
  bool isComplex() const { return V1.getInt() == Complex; }
  bool isAggregate() const { return V1.getInt() == Aggregate; }
  bool isIgnored() const { return isScalar() && !getScalarVal(); }

  bool isVolatileQualified() const { return V2.getInt(); }

  /// Return the mlir::Value of this scalar value.
  mlir::Value getScalarVal() const {
    assert(isScalar() && "Not a scalar!");
    return V1.getPointer();
  }

  /// Return the real/imag components of this complex value.
  std::pair<mlir::Value, mlir::Value> getComplexVal() const {
    assert(0 && "not implemented");
    return {};
  }

  /// Return the mlir::Value of the address of the aggregate.
  Address getAggregateAddress() const {
    assert(isAggregate() && "Not an aggregate!");
    auto align = reinterpret_cast<uintptr_t>(V2.getPointer().get<int *>()) >>
                 AggAlignShift;
    return Address(V1.getPointer(), ElementType,
                   clang::CharUnits::fromQuantity(align));
  }

  mlir::Value getAggregatePointer() const {
    assert(isAggregate() && "Not an aggregate!");
    return V1.getPointer();
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
  // FIXME: Aggregate rvalues need to retain information about whether they are
  // volatile or not. Remove default to find all places that probably get this
  // wrong.
  static RValue getAggregate(Address addr, bool isVolatile = false) {
    RValue ER;
    ER.V1.setPointer(addr.getPointer());
    ER.V1.setInt(Aggregate);
    ER.ElementType = addr.getElementType();

    auto align = static_cast<uintptr_t>(addr.getAlignment().getQuantity());
    ER.V2.setPointer(reinterpret_cast<int *>(align << AggAlignShift));
    ER.V2.setInt(isVolatile);
    return ER;
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

  // LValue is non-gc'able for any reason, including being a parameter or local
  // variable.
  bool NonGC : 1;

  // This flag shows if a nontemporal load/stores should be used when accessing
  // this lvalue.
  bool Nontemporal : 1;

private:
  void Initialize(clang::QualType Type, clang::Qualifiers Quals,
                  clang::CharUnits Alignment, LValueBaseInfo BaseInfo) {
    assert((!Alignment.isZero() || Type->isIncompleteType()) &&
           "initializing l-value with zero alignment!");
    if (isGlobalReg())
      assert(ElementType == nullptr && "Global reg does not store elem type");

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
    this->NonGC = false;
    this->Nontemporal = false;
  }

  // The alignment to use when accessing this lvalue. (For vector elements,
  // this is the alignment of the whole vector)
  unsigned Alignment;
  mlir::Value V;
  mlir::Type ElementType;
  LValueBaseInfo BaseInfo;
  const CIRGenBitFieldInfo *BitFieldInfo{0};

public:
  bool isSimple() const { return LVType == Simple; }
  bool isVectorElt() const { return LVType == VectorElt; }
  bool isBitField() const { return LVType == BitField; }
  bool isExtVectorElt() const { return LVType == ExtVectorElt; }
  bool isGlobalReg() const { return LVType == GlobalReg; }
  bool isMatrixElt() const { return LVType == MatrixElt; }

  bool isVolatileQualified() const { return Quals.hasVolatile(); }

  unsigned getVRQualifiers() const {
    return Quals.getCVRQualifiers() & ~clang::Qualifiers::Const;
  }

  bool isNonGC() const { return NonGC; }
  void setNonGC(bool Value) { NonGC = Value; }

  bool isNontemporal() const { return Nontemporal; }

  bool isObjCWeak() const {
    return Quals.getObjCGCAttr() == clang::Qualifiers::Weak;
  }
  bool isObjCStrong() const {
    return Quals.getObjCGCAttr() == clang::Qualifiers::Strong;
  }

  bool isVolatile() const { return Quals.hasVolatile(); }

  clang::QualType getType() const { return Type; }

  mlir::Value getPointer() const { return V; }

  clang::CharUnits getAlignment() const {
    return clang::CharUnits::fromQuantity(Alignment);
  }

  Address getAddress() const {
    return Address(getPointer(), ElementType, getAlignment());
  }

  void setAddress(Address address) {
    assert(isSimple());
    V = address.getPointer();
    ElementType = address.getElementType();
    Alignment = address.getAlignment().getQuantity();
    // TODO(cir): IsKnownNonNull = address.isKnownNonNull();
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

  // bitfield lvalue
  Address getBitFieldAddress() const {
    return Address(getBitFieldPointer(), ElementType, getAlignment());
  }

  mlir::Value getBitFieldPointer() const {
    assert(isBitField());
    return V;
  }

  const CIRGenBitFieldInfo &getBitFieldInfo() const {
    assert(isBitField());
    return *BitFieldInfo;
  }

  /// Create a new object to represent a bit-field access.
  ///
  /// \param Addr - The base address of the bit-field sequence this
  /// bit-field refers to.
  /// \param Info - The information describing how to perform the bit-field
  /// access.
  static LValue MakeBitfield(Address Addr, const CIRGenBitFieldInfo &Info,
                             clang::QualType type, LValueBaseInfo BaseInfo) {
    LValue R;
    R.LVType = BitField;
    R.V = Addr.getPointer();
    R.ElementType = Addr.getElementType();
    R.BitFieldInfo = &Info;
    R.Initialize(type, type.getQualifiers(), Addr.getAlignment(), BaseInfo);
    return R;
  }
};

/// An aggregate value slot.
class AggValueSlot {
  /// The address.
  Address Addr;

  // Qualifiers
  clang::Qualifiers Quals;

  /// This is set to true if some external code is responsible for setting up a
  /// destructor for the slot.  Otherwise the code which constructs it should
  /// push the appropriate cleanup.
  bool DestructedFlag : 1;

  /// This is set to true if writing to the memory in the slot might require
  /// calling an appropriate Objective-C GC barrier.  The exact interaction here
  /// is unnecessarily mysterious.
  bool ObjCGCFlag : 1;

  /// This is set to true if the memory in the slot is known to be zero before
  /// the assignment into it.  This means that zero fields don't need to be set.
  bool ZeroedFlag : 1;

  /// This is set to true if the slot might be aliased and it's not undefined
  /// behavior to access it through such an alias.  Note that it's always
  /// undefined behavior to access a C++ object that's under construction
  /// through an alias derived from outside the construction process.
  ///
  /// This flag controls whether calls that produce the aggregate
  /// value may be evaluated directly into the slot, or whether they
  /// must be evaluated into an unaliased temporary and then memcpy'ed
  /// over.  Since it's invalid in general to memcpy a non-POD C++
  /// object, it's important that this flag never be set when
  /// evaluating an expression which constructs such an object.
  bool AliasedFlag : 1;

  /// This is set to true if the tail padding of this slot might overlap
  /// another object that may have already been initialized (and whose
  /// value must be preserved by this initialization). If so, we may only
  /// store up to the dsize of the type. Otherwise we can widen stores to
  /// the size of the type.
  bool OverlapFlag : 1;

  /// If is set to true, sanitizer checks are already generated for this address
  /// or not required. For instance, if this address represents an object
  /// created in 'new' expression, sanitizer checks for memory is made as a part
  /// of 'operator new' emission and object constructor should not generate
  /// them.
  bool SanitizerCheckedFlag : 1;

  AggValueSlot(Address Addr, clang::Qualifiers Quals, bool DestructedFlag,
               bool ObjCGCFlag, bool ZeroedFlag, bool AliasedFlag,
               bool OverlapFlag, bool SanitizerCheckedFlag)
      : Addr(Addr), Quals(Quals), DestructedFlag(DestructedFlag),
        ObjCGCFlag(ObjCGCFlag), ZeroedFlag(ZeroedFlag),
        AliasedFlag(AliasedFlag), OverlapFlag(OverlapFlag),
        SanitizerCheckedFlag(SanitizerCheckedFlag) {}

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

  IsDestructed_t isExternallyDestructed() const {
    return IsDestructed_t(DestructedFlag);
  }
  void setExternallyDestructed(bool destructed = true) {
    DestructedFlag = destructed;
  }

  clang::Qualifiers getQualifiers() const { return Quals; }

  bool isVolatile() const { return Quals.hasVolatile(); }

  Address getAddress() const { return Addr; }

  bool isIgnored() const { return !Addr.isValid(); }

  mlir::Value getPointer() const { return Addr.getPointer(); }

  Overlap_t mayOverlap() const { return Overlap_t(OverlapFlag); }

  bool isSanitizerChecked() const { return SanitizerCheckedFlag; }

  IsZeroed_t isZeroed() const { return IsZeroed_t(ZeroedFlag); }
  void setZeroed(bool V = true) { ZeroedFlag = V; }

  NeedsGCBarriers_t requiresGCollection() const {
    return NeedsGCBarriers_t(ObjCGCFlag);
  }

  IsAliased_t isPotentiallyAliased() const { return IsAliased_t(AliasedFlag); }

  RValue asRValue() const {
    if (isIgnored()) {
      return RValue::getIgnored();
    } else {
      return RValue::getAggregate(getAddress(), isVolatile());
    }
  }

  /// Get the preferred size to use when storing a value to this slot. This
  /// is the type size unless that might overlap another object, in which
  /// case it's the dsize.
  clang::CharUnits getPreferredSize(clang::ASTContext &Ctx,
                                    clang::QualType Type) {
    return mayOverlap() ? Ctx.getTypeInfoDataSizeInChars(Type).Width
                        : Ctx.getTypeSizeInChars(Type);
  }
};

} // namespace cir

#endif
