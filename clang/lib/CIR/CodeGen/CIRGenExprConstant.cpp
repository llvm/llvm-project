//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Constant Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CIRGenCXXABI.h"
#include "CIRGenConstantEmitter.h"
#include "CIRGenModule.h"
#include "CIRGenRecordLayout.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
#include <iterator>

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
//                            ConstantAggregateBuilder
//===----------------------------------------------------------------------===//

namespace {
class ConstExprEmitter;

static mlir::TypedAttr computePadding(CIRGenModule &cgm, CharUnits size) {
  mlir::Type eltTy = cgm.UCharTy;
  clang::CharUnits::QuantityType arSize = size.getQuantity();
  CIRGenBuilderTy &bld = cgm.getBuilder();
  if (size > CharUnits::One()) {
    SmallVector<mlir::Attribute> elts(arSize, cir::ZeroAttr::get(eltTy));
    return bld.getConstArray(mlir::ArrayAttr::get(bld.getContext(), elts),
                             cir::ArrayType::get(eltTy, arSize));
  }

  return cir::ZeroAttr::get(eltTy);
}

static mlir::Attribute
emitArrayConstant(CIRGenModule &cgm, mlir::Type desiredType,
                  mlir::Type commonElementType, unsigned arrayBound,
                  SmallVectorImpl<mlir::TypedAttr> &elements,
                  mlir::TypedAttr filler);

struct ConstantAggregateBuilderUtils {
  CIRGenModule &cgm;
  cir::CIRDataLayout dataLayout;

  ConstantAggregateBuilderUtils(CIRGenModule &cgm)
      : cgm(cgm), dataLayout{cgm.getModule()} {}

  CharUnits getAlignment(const mlir::TypedAttr c) const {
    return CharUnits::fromQuantity(
        dataLayout.getAlignment(c.getType(), /*useABIAlign=*/true));
  }

  CharUnits getSize(mlir::Type ty) const {
    return CharUnits::fromQuantity(dataLayout.getTypeAllocSize(ty));
  }

  CharUnits getSize(const mlir::TypedAttr c) const {
    return getSize(c.getType());
  }

  mlir::TypedAttr getPadding(CharUnits size) const {
    return computePadding(cgm, size);
  }
};

/// Incremental builder for an mlir::TypedAttr holding a record or array
/// constant.
class ConstantAggregateBuilder : private ConstantAggregateBuilderUtils {
  struct Element {
    Element(mlir::TypedAttr element, CharUnits offset)
        : element(element), offset(offset) {}

    mlir::TypedAttr element;
    /// Describes the offset of `element` within the constant.
    CharUnits offset;
  };
  /// The elements of the constant. The elements are kept in increasing offset
  /// order, and we ensure that there is no overlap:
  /// elements.offset[i+1] >= elements.offset[i] + getSize(elements.element[i])
  ///
  /// This may contain explicit padding elements (in order to create a
  /// natural layout), but need not. Gaps between elements are implicitly
  /// considered to be filled with undef.
  llvm::SmallVector<Element, 32> elements;

  /// The size of the constant (the maximum end offset of any added element).
  /// May be larger than the end of elems.back() if we split the last element
  /// and removed some trailing undefs.
  CharUnits size = CharUnits::Zero();

  /// This is true only if laying out elems in order as the elements of a
  /// non-packed LLVM struct will give the correct layout.
  bool naturalLayout = true;

  static mlir::Attribute buildFrom(CIRGenModule &cgm, ArrayRef<Element> elems,
                                   CharUnits startOffset, CharUnits size,
                                   bool naturalLayout, mlir::Type desiredTy,
                                   bool allowOversized);

public:
  ConstantAggregateBuilder(CIRGenModule &cgm)
      : ConstantAggregateBuilderUtils(cgm) {}

  /// Update or overwrite the value starting at \p offset with \c c.
  ///
  /// \param allowOverwrite If \c true, this constant might overwrite (part of)
  ///        a constant that has already been added. This flag is only used to
  ///        detect bugs.
  bool add(mlir::TypedAttr typedAttr, CharUnits offset, bool allowOverwrite);

  /// Update or overwrite the bits starting at \p offsetInBits with \p bits.
  bool addBits(llvm::APInt bits, uint64_t offsetInBits, bool allowOverwrite);

  /// Produce a constant representing the entire accumulated value, ideally of
  /// the specified type. If \p allowOversized, the constant might be larger
  /// than implied by \p desiredTy (eg, if there is a flexible array member).
  /// Otherwise, the constant will be of exactly the same size as \p desiredTy
  /// even if we can't represent it as that type.
  mlir::Attribute build(mlir::Type desiredTy, bool allowOversized) const {
    return buildFrom(cgm, elements, CharUnits::Zero(), size, naturalLayout,
                     desiredTy, allowOversized);
  }
};

template <typename Container, typename Range = std::initializer_list<
                                  typename Container::value_type>>
static void replace(Container &c, size_t beginOff, size_t endOff, Range vals) {
  assert(beginOff <= endOff && "invalid replacement range");
  llvm::replace(c, c.begin() + beginOff, c.begin() + endOff, vals);
}

bool ConstantAggregateBuilder::add(mlir::TypedAttr typedAttr, CharUnits offset,
                                   bool allowOverwrite) {
  // Common case: appending to a layout.
  if (offset >= size) {
    CharUnits align = getAlignment(typedAttr);
    CharUnits alignedSize = size.alignTo(align);
    if (alignedSize > offset || offset.alignTo(align) != offset) {
      naturalLayout = false;
    } else if (alignedSize < offset) {
      elements.emplace_back(getPadding(offset - size), size);
    }
    elements.emplace_back(typedAttr, offset);
    size = offset + getSize(typedAttr);
    return true;
  }

  // Uncommon case: constant overlaps what we've already created.
  cgm.errorNYI("overlapping constants");
  return false;
}

mlir::Attribute
ConstantAggregateBuilder::buildFrom(CIRGenModule &cgm, ArrayRef<Element> elems,
                                    CharUnits startOffset, CharUnits size,
                                    bool naturalLayout, mlir::Type desiredTy,
                                    bool allowOversized) {
  ConstantAggregateBuilderUtils utils(cgm);

  if (elems.empty())
    return cir::UndefAttr::get(desiredTy);

  // If we want an array type, see if all the elements are the same type and
  // appropriately spaced.
  if (mlir::isa<cir::ArrayType>(desiredTy)) {
    cgm.errorNYI("array aggregate constants");
    return {};
  }

  // The size of the constant we plan to generate. This is usually just the size
  // of the initialized type, but in AllowOversized mode (i.e. flexible array
  // init), it can be larger.
  CharUnits desiredSize = utils.getSize(desiredTy);
  if (size > desiredSize) {
    assert(allowOversized && "elems are oversized");
    desiredSize = size;
  }

  // The natural alignment of an unpacked CIR record with the given elements.
  CharUnits align = CharUnits::One();
  for (auto [e, offset] : elems)
    align = std::max(align, utils.getAlignment(e));

  // The natural size of an unpacked LLVM struct with the given elements.
  CharUnits alignedSize = size.alignTo(align);

  bool packed = false;
  bool padded = false;

  llvm::SmallVector<mlir::Attribute, 32> unpackedElems;
  if (desiredSize < alignedSize || desiredSize.alignTo(align) != desiredSize) {
    naturalLayout = false;
    packed = true;
  } else {
    // The natural layout would be too small. Add padding to fix it. (This
    // is ignored if we choose a packed layout.)
    unpackedElems.reserve(elems.size() + 1);
    llvm::transform(elems, std::back_inserter(unpackedElems),
                    std::mem_fn(&Element::element));
    if (desiredSize > alignedSize)
      unpackedElems.push_back(utils.getPadding(desiredSize - size));
  }

  // If we don't have a natural layout, insert padding as necessary.
  // As we go, double-check to see if we can actually just emit Elems
  // as a non-packed record and do so opportunistically if possible.
  llvm::SmallVector<mlir::Attribute, 32> packedElems;
  packedElems.reserve(elems.size());
  if (!naturalLayout) {
    CharUnits sizeSoFar = CharUnits::Zero();
    for (auto [element, offset] : elems) {
      CharUnits align = utils.getAlignment(element);
      CharUnits naturalOffset = sizeSoFar.alignTo(align);
      CharUnits desiredOffset = offset - startOffset;
      assert(desiredOffset >= sizeSoFar && "elements out of order");

      if (desiredOffset != naturalOffset)
        packed = true;
      if (desiredOffset != sizeSoFar)
        packedElems.push_back(utils.getPadding(desiredOffset - sizeSoFar));
      packedElems.push_back(element);
      sizeSoFar = desiredOffset + utils.getSize(element);
    }
    // If we're using the packed layout, pad it out to the desired size if
    // necessary.
    if (packed) {
      assert(sizeSoFar <= desiredSize &&
             "requested size is too small for contents");

      if (sizeSoFar < desiredSize)
        packedElems.push_back(utils.getPadding(desiredSize - sizeSoFar));
    }
  }

  CIRGenBuilderTy &builder = cgm.getBuilder();
  auto arrAttr = mlir::ArrayAttr::get(builder.getContext(),
                                      packed ? packedElems : unpackedElems);

  cir::RecordType recordType = builder.getCompleteRecordType(arrAttr, packed);
  if (auto desired = mlir::dyn_cast<cir::RecordType>(desiredTy))
    if (desired.isLayoutIdentical(recordType))
      recordType = desired;

  return builder.getConstRecordOrZeroAttr(arrAttr, packed, padded, recordType);
}

//===----------------------------------------------------------------------===//
//                            ConstRecordBuilder
//===----------------------------------------------------------------------===//

class ConstRecordBuilder {
  CIRGenModule &cgm;
  ConstantEmitter &emitter;
  ConstantAggregateBuilder &builder;
  CharUnits startOffset;

public:
  static mlir::Attribute buildRecord(ConstantEmitter &emitter,
                                     InitListExpr *ile, QualType valTy);
  static mlir::Attribute buildRecord(ConstantEmitter &emitter,
                                     const APValue &value, QualType valTy);
  static bool updateRecord(ConstantEmitter &emitter,
                           ConstantAggregateBuilder &constant, CharUnits offset,
                           InitListExpr *updater);

private:
  ConstRecordBuilder(ConstantEmitter &emitter,
                     ConstantAggregateBuilder &builder, CharUnits startOffset)
      : cgm(emitter.cgm), emitter(emitter), builder(builder),
        startOffset(startOffset) {}

  bool appendField(const FieldDecl *field, uint64_t fieldOffset,
                   mlir::TypedAttr initCst, bool allowOverwrite = false);

  bool appendBytes(CharUnits fieldOffsetInChars, mlir::TypedAttr initCst,
                   bool allowOverwrite = false);

  bool build(InitListExpr *ile, bool allowOverwrite);
  bool build(const APValue &val, const RecordDecl *rd, bool isPrimaryBase,
             const CXXRecordDecl *vTableClass, CharUnits baseOffset);

  mlir::Attribute finalize(QualType ty);
};

bool ConstRecordBuilder::appendField(const FieldDecl *field,
                                     uint64_t fieldOffset,
                                     mlir::TypedAttr initCst,
                                     bool allowOverwrite) {
  const ASTContext &astContext = cgm.getASTContext();

  CharUnits fieldOffsetInChars = astContext.toCharUnitsFromBits(fieldOffset);

  return appendBytes(fieldOffsetInChars, initCst, allowOverwrite);
}

bool ConstRecordBuilder::appendBytes(CharUnits fieldOffsetInChars,
                                     mlir::TypedAttr initCst,
                                     bool allowOverwrite) {
  return builder.add(initCst, startOffset + fieldOffsetInChars, allowOverwrite);
}

bool ConstRecordBuilder::build(InitListExpr *ile, bool allowOverwrite) {
  RecordDecl *rd = ile->getType()
                       ->castAs<clang::RecordType>()
                       ->getOriginalDecl()
                       ->getDefinitionOrSelf();
  const ASTRecordLayout &layout = cgm.getASTContext().getASTRecordLayout(rd);

  // Bail out if we have base classes. We could support these, but they only
  // arise in C++1z where we will have already constant folded most interesting
  // cases. FIXME: There are still a few more cases we can handle this way.
  if (auto *cxxrd = dyn_cast<CXXRecordDecl>(rd))
    if (cxxrd->getNumBases())
      return false;

  if (cgm.shouldZeroInitPadding()) {
    assert(!cir::MissingFeatures::recordZeroInitPadding());
    cgm.errorNYI(rd->getSourceRange(), "zero init padding");
    return false;
  }

  unsigned elementNo = 0;
  for (auto [index, field] : llvm::enumerate(rd->fields())) {

    // If this is a union, skip all the fields that aren't being initialized.
    if (rd->isUnion() &&
        !declaresSameEntity(ile->getInitializedFieldInUnion(), field))
      continue;

    // Don't emit anonymous bitfields.
    if (field->isUnnamedBitField())
      continue;

    // Get the initializer.  A record can include fields without initializers,
    // we just use explicit null values for them.
    Expr *init = nullptr;
    if (elementNo < ile->getNumInits())
      init = ile->getInit(elementNo++);
    if (isa_and_nonnull<NoInitExpr>(init))
      continue;

    // Zero-sized fields are not emitted, but their initializers may still
    // prevent emission of this record as a constant.
    if (field->isZeroSize(cgm.getASTContext())) {
      if (init->HasSideEffects(cgm.getASTContext()))
        return false;
      continue;
    }

    assert(!cir::MissingFeatures::recordZeroInitPadding());

    // When emitting a DesignatedInitUpdateExpr, a nested InitListExpr
    // represents additional overwriting of our current constant value, and not
    // a new constant to emit independently.
    if (allowOverwrite &&
        (field->getType()->isArrayType() || field->getType()->isRecordType())) {
      cgm.errorNYI(field->getSourceRange(), "designated init lists");
      return false;
    }

    mlir::TypedAttr eltInit;
    if (init)
      eltInit = mlir::cast<mlir::TypedAttr>(
          emitter.tryEmitPrivateForMemory(init, field->getType()));
    else
      eltInit = mlir::cast<mlir::TypedAttr>(emitter.emitNullForMemory(
          cgm.getLoc(ile->getSourceRange()), field->getType()));

    if (!eltInit)
      return false;

    if (!field->isBitField()) {
      // Handle non-bitfield members.
      if (!appendField(field, layout.getFieldOffset(index), eltInit,
                       allowOverwrite))
        return false;
      // After emitting a non-empty field with [[no_unique_address]], we may
      // need to overwrite its tail padding.
      if (field->hasAttr<NoUniqueAddressAttr>())
        allowOverwrite = true;
    } else {
      // Otherwise we have a bitfield.
      if (auto constInt = dyn_cast<cir::IntAttr>(eltInit)) {
        assert(!cir::MissingFeatures::bitfields());
        cgm.errorNYI(field->getSourceRange(), "bitfields");
      }
      // We are trying to initialize a bitfield with a non-trivial constant,
      // this must require run-time code.
      return false;
    }
  }

  assert(!cir::MissingFeatures::recordZeroInitPadding());
  return true;
}

namespace {
struct BaseInfo {
  BaseInfo(const CXXRecordDecl *decl, CharUnits offset, unsigned index)
      : decl(decl), offset(offset), index(index) {}

  const CXXRecordDecl *decl;
  CharUnits offset;
  unsigned index;

  bool operator<(const BaseInfo &o) const { return offset < o.offset; }
};
} // namespace

bool ConstRecordBuilder::build(const APValue &val, const RecordDecl *rd,
                               bool isPrimaryBase,
                               const CXXRecordDecl *vTableClass,
                               CharUnits offset) {
  const ASTRecordLayout &layout = cgm.getASTContext().getASTRecordLayout(rd);
  if (const CXXRecordDecl *cd = dyn_cast<CXXRecordDecl>(rd)) {
    // Add a vtable pointer, if we need one and it hasn't already been added.
    if (layout.hasOwnVFPtr()) {
      CIRGenBuilderTy &builder = cgm.getBuilder();
      cir::GlobalOp vtable =
          cgm.getCXXABI().getAddrOfVTable(vTableClass, CharUnits());
      clang::VTableLayout::AddressPointLocation addressPoint =
          cgm.getItaniumVTableContext()
              .getVTableLayout(vTableClass)
              .getAddressPoint(BaseSubobject(cd, offset));
      assert(!cir::MissingFeatures::addressPointerAuthInfo());
      mlir::ArrayAttr indices = builder.getArrayAttr({
          builder.getI32IntegerAttr(addressPoint.VTableIndex),
          builder.getI32IntegerAttr(addressPoint.AddressPointIndex),
      });
      cir::GlobalViewAttr vtableInit =
          cgm.getBuilder().getGlobalViewAttr(vtable, indices);
      if (!appendBytes(offset, vtableInit))
        return false;
    }

    // Accumulate and sort bases, in order to visit them in address order, which
    // may not be the same as declaration order.
    SmallVector<BaseInfo> bases;
    bases.reserve(cd->getNumBases());
    for (auto [index, base] : llvm::enumerate(cd->bases())) {
      assert(!base.isVirtual() && "should not have virtual bases here");
      const CXXRecordDecl *bd = base.getType()->getAsCXXRecordDecl();
      CharUnits baseOffset = layout.getBaseClassOffset(bd);
      bases.push_back(BaseInfo(bd, baseOffset, index));
    }
#ifdef EXPENSIVE_CHECKS
    assert(llvm::is_sorted(bases) && "bases not sorted by offset");
#endif

    for (BaseInfo &base : bases) {
      bool isPrimaryBase = layout.getPrimaryBase() == base.decl;
      build(val.getStructBase(base.index), base.decl, isPrimaryBase,
            vTableClass, offset + base.offset);
    }
  }

  uint64_t offsetBits = cgm.getASTContext().toBits(offset);

  bool allowOverwrite = false;
  for (auto [index, field] : llvm::enumerate(rd->fields())) {
    // If this is a union, skip all the fields that aren't being initialized.
    if (rd->isUnion() && !declaresSameEntity(val.getUnionField(), field))
      continue;

    // Don't emit anonymous bitfields or zero-sized fields.
    if (field->isUnnamedBitField() || field->isZeroSize(cgm.getASTContext()))
      continue;

    // Emit the value of the initializer.
    const APValue &fieldValue =
        rd->isUnion() ? val.getUnionValue() : val.getStructField(index);
    mlir::TypedAttr eltInit = mlir::cast<mlir::TypedAttr>(
        emitter.tryEmitPrivateForMemory(fieldValue, field->getType()));
    if (!eltInit)
      return false;

    if (!field->isBitField()) {
      // Handle non-bitfield members.
      if (!appendField(field, layout.getFieldOffset(index) + offsetBits,
                       eltInit, allowOverwrite))
        return false;
      // After emitting a non-empty field with [[no_unique_address]], we may
      // need to overwrite its tail padding.
      if (field->hasAttr<NoUniqueAddressAttr>())
        allowOverwrite = true;
    } else {
      assert(!cir::MissingFeatures::bitfields());
      cgm.errorNYI(field->getSourceRange(), "bitfields");
    }
  }

  return true;
}

mlir::Attribute ConstRecordBuilder::finalize(QualType type) {
  type = type.getNonReferenceType();
  RecordDecl *rd = type->castAs<clang::RecordType>()
                       ->getOriginalDecl()
                       ->getDefinitionOrSelf();
  mlir::Type valTy = cgm.convertType(type);
  return builder.build(valTy, rd->hasFlexibleArrayMember());
}

mlir::Attribute ConstRecordBuilder::buildRecord(ConstantEmitter &emitter,
                                                InitListExpr *ile,
                                                QualType valTy) {
  ConstantAggregateBuilder constant(emitter.cgm);
  ConstRecordBuilder builder(emitter, constant, CharUnits::Zero());

  if (!builder.build(ile, /*allowOverwrite*/ false))
    return nullptr;

  return builder.finalize(valTy);
}

mlir::Attribute ConstRecordBuilder::buildRecord(ConstantEmitter &emitter,
                                                const APValue &val,
                                                QualType valTy) {
  ConstantAggregateBuilder constant(emitter.cgm);
  ConstRecordBuilder builder(emitter, constant, CharUnits::Zero());

  const RecordDecl *rd = valTy->castAs<clang::RecordType>()
                             ->getOriginalDecl()
                             ->getDefinitionOrSelf();
  const CXXRecordDecl *cd = dyn_cast<CXXRecordDecl>(rd);
  if (!builder.build(val, rd, false, cd, CharUnits::Zero()))
    return nullptr;

  return builder.finalize(valTy);
}

bool ConstRecordBuilder::updateRecord(ConstantEmitter &emitter,
                                      ConstantAggregateBuilder &constant,
                                      CharUnits offset, InitListExpr *updater) {
  return ConstRecordBuilder(emitter, constant, offset)
      .build(updater, /*allowOverwrite*/ true);
}

//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//

// This class only needs to handle arrays, structs and unions.
//
// In LLVM codegen, when outside C++11 mode, those types are not constant
// folded, while all other types are handled by constant folding.
//
// In CIR codegen, instead of folding things here, we should defer that work
// to MLIR: do not attempt to do much here.
class ConstExprEmitter
    : public StmtVisitor<ConstExprEmitter, mlir::Attribute, QualType> {
  CIRGenModule &cgm;
  LLVM_ATTRIBUTE_UNUSED ConstantEmitter &emitter;

public:
  ConstExprEmitter(ConstantEmitter &emitter)
      : cgm(emitter.cgm), emitter(emitter) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Attribute VisitStmt(Stmt *s, QualType t) { return {}; }

  mlir::Attribute VisitConstantExpr(ConstantExpr *ce, QualType t) {
    if (mlir::Attribute result = emitter.tryEmitConstantExpr(ce))
      return result;
    return Visit(ce->getSubExpr(), t);
  }

  mlir::Attribute VisitParenExpr(ParenExpr *pe, QualType t) {
    return Visit(pe->getSubExpr(), t);
  }

  mlir::Attribute
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *pe,
                                    QualType t) {
    return Visit(pe->getReplacement(), t);
  }

  mlir::Attribute VisitGenericSelectionExpr(GenericSelectionExpr *ge,
                                            QualType t) {
    return Visit(ge->getResultExpr(), t);
  }

  mlir::Attribute VisitChooseExpr(ChooseExpr *ce, QualType t) {
    return Visit(ce->getChosenSubExpr(), t);
  }

  mlir::Attribute VisitCompoundLiteralExpr(CompoundLiteralExpr *e, QualType t) {
    return Visit(e->getInitializer(), t);
  }

  mlir::Attribute VisitCastExpr(CastExpr *e, QualType destType) {
    if (isa<ExplicitCastExpr>(e))
      cgm.errorNYI(e->getBeginLoc(),
                   "ConstExprEmitter::VisitCastExpr explicit cast");

    Expr *subExpr = e->getSubExpr();

    switch (e->getCastKind()) {
    case CK_ToUnion:
    case CK_AddressSpaceConversion:
    case CK_ReinterpretMemberPointer:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer:
      cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitCastExpr");
      return {};

    case CK_LValueToRValue:
    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_ConstructorConversion:
      return Visit(subExpr, destType);

    case CK_IntToOCLSampler:
      llvm_unreachable("global sampler variables are not generated");

    case CK_Dependent:
      llvm_unreachable("saw dependent cast!");

    case CK_BuiltinFnToFnPtr:
      llvm_unreachable("builtin functions are handled elsewhere");

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_ARCProduceObject:
    case CK_ARCConsumeObject:
    case CK_ARCReclaimReturnedObject:
    case CK_ARCExtendBlockObject:
    case CK_CopyAndAutoreleaseBlockObject:
      return {};

    // These don't need to be handled here because Evaluate knows how to
    // evaluate them in the cases where they can be folded.
    case CK_BitCast:
    case CK_ToVoid:
    case CK_Dynamic:
    case CK_LValueBitCast:
    case CK_LValueToRValueBitCast:
    case CK_NullToMemberPointer:
    case CK_UserDefinedConversion:
    case CK_CPointerToObjCPointerCast:
    case CK_BlockPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_MemberPointerToBoolean:
    case CK_VectorSplat:
    case CK_FloatingRealToComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingComplexToBoolean:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToIntegralComplex:
    case CK_IntegralRealToComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralComplexToBoolean:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToFloatingComplex:
    case CK_PointerToIntegral:
    case CK_PointerToBoolean:
    case CK_NullToPointer:
    case CK_IntegralCast:
    case CK_BooleanToSignedIntegral:
    case CK_IntegralToPointer:
    case CK_IntegralToBoolean:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingToBoolean:
    case CK_FloatingCast:
    case CK_FloatingToFixedPoint:
    case CK_FixedPointToFloating:
    case CK_FixedPointCast:
    case CK_FixedPointToBoolean:
    case CK_FixedPointToIntegral:
    case CK_IntegralToFixedPoint:
    case CK_ZeroToOCLOpaqueType:
    case CK_MatrixCast:
    case CK_HLSLArrayRValue:
    case CK_HLSLVectorTruncation:
    case CK_HLSLElementwiseCast:
    case CK_HLSLAggregateSplatCast:
      return {};
    }
    llvm_unreachable("Invalid CastKind");
  }

  mlir::Attribute VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die, QualType t) {
    cgm.errorNYI(die->getBeginLoc(),
                 "ConstExprEmitter::VisitCXXDefaultInitExpr");
    return {};
  }

  mlir::Attribute VisitExprWithCleanups(ExprWithCleanups *e, QualType t) {
    // Since this about constant emission no need to wrap this under a scope.
    return Visit(e->getSubExpr(), t);
  }

  mlir::Attribute VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *e,
                                                QualType t) {
    return Visit(e->getSubExpr(), t);
  }

  mlir::Attribute VisitImplicitValueInitExpr(ImplicitValueInitExpr *e,
                                             QualType t) {
    cgm.errorNYI(e->getBeginLoc(),
                 "ConstExprEmitter::VisitImplicitValueInitExpr");
    return {};
  }

  mlir::Attribute VisitInitListExpr(InitListExpr *ile, QualType t) {
    if (ile->isTransparent())
      return Visit(ile->getInit(0), t);

    if (ile->getType()->isArrayType()) {
      // If we return null here, the non-constant initializer will take care of
      // it, but we would prefer to handle it here.
      assert(!cir::MissingFeatures::constEmitterArrayILE());
      return {};
    }

    if (ile->getType()->isRecordType()) {
      return ConstRecordBuilder::buildRecord(emitter, ile, t);
    }

    if (ile->getType()->isVectorType()) {
      // If we return null here, the non-constant initializer will take care of
      // it, but we would prefer to handle it here.
      assert(!cir::MissingFeatures::constEmitterVectorILE());
      return {};
    }

    return {};
  }

  mlir::Attribute VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *e,
                                                QualType destType) {
    mlir::Attribute c = Visit(e->getBase(), destType);
    if (!c)
      return {};

    cgm.errorNYI(e->getBeginLoc(),
                 "ConstExprEmitter::VisitDesignatedInitUpdateExpr");
    return {};
  }

  mlir::Attribute VisitCXXConstructExpr(CXXConstructExpr *e, QualType ty) {
    cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitCXXConstructExpr");
    return {};
  }

  mlir::Attribute VisitStringLiteral(StringLiteral *e, QualType t) {
    // This is a string literal initializing an array in an initializer.
    return cgm.getConstantArrayFromStringLiteral(e);
  }

  mlir::Attribute VisitObjCEncodeExpr(ObjCEncodeExpr *e, QualType t) {
    cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitObjCEncodeExpr");
    return {};
  }

  mlir::Attribute VisitUnaryExtension(const UnaryOperator *e, QualType t) {
    return Visit(e->getSubExpr(), t);
  }

  // Utility methods
  mlir::Type convertType(QualType t) { return cgm.convertType(t); }
};

// TODO(cir): this can be shared with LLVM's codegen
static QualType getNonMemoryType(CIRGenModule &cgm, QualType type) {
  if (const auto *at = type->getAs<AtomicType>()) {
    return cgm.getASTContext().getQualifiedType(at->getValueType(),
                                                type.getQualifiers());
  }
  return type;
}

static mlir::Attribute
emitArrayConstant(CIRGenModule &cgm, mlir::Type desiredType,
                  mlir::Type commonElementType, unsigned arrayBound,
                  SmallVectorImpl<mlir::TypedAttr> &elements,
                  mlir::TypedAttr filler) {
  CIRGenBuilderTy &builder = cgm.getBuilder();

  unsigned nonzeroLength = arrayBound;
  if (elements.size() < nonzeroLength && builder.isNullValue(filler))
    nonzeroLength = elements.size();

  if (nonzeroLength == elements.size()) {
    while (nonzeroLength > 0 &&
           builder.isNullValue(elements[nonzeroLength - 1]))
      --nonzeroLength;
  }

  if (nonzeroLength == 0)
    return cir::ZeroAttr::get(desiredType);

  const unsigned trailingZeroes = arrayBound - nonzeroLength;

  // Add a zeroinitializer array filler if we have lots of trailing zeroes.
  if (trailingZeroes >= 8) {
    assert(elements.size() >= nonzeroLength &&
           "missing initializer for non-zero element");

    if (commonElementType && nonzeroLength >= 8) {
      // If all the elements had the same type up to the trailing zeroes and
      // there are eight or more nonzero elements, emit a struct of two arrays
      // (the nonzero data and the zeroinitializer).
      SmallVector<mlir::Attribute> eles;
      eles.reserve(nonzeroLength);
      for (const auto &element : elements)
        eles.push_back(element);
      auto initial = cir::ConstArrayAttr::get(
          cir::ArrayType::get(commonElementType, nonzeroLength),
          mlir::ArrayAttr::get(builder.getContext(), eles));
      elements.resize(2);
      elements[0] = initial;
    } else {
      // Otherwise, emit a struct with individual elements for each nonzero
      // initializer, followed by a zeroinitializer array filler.
      elements.resize(nonzeroLength + 1);
    }

    mlir::Type fillerType =
        commonElementType
            ? commonElementType
            : mlir::cast<cir::ArrayType>(desiredType).getElementType();
    fillerType = cir::ArrayType::get(fillerType, trailingZeroes);
    elements.back() = cir::ZeroAttr::get(fillerType);
    commonElementType = nullptr;
  } else if (elements.size() != arrayBound) {
    elements.resize(arrayBound, filler);

    if (filler.getType() != commonElementType)
      commonElementType = {};
  }

  if (commonElementType) {
    SmallVector<mlir::Attribute> eles;
    eles.reserve(elements.size());

    for (const auto &element : elements)
      eles.push_back(element);

    return cir::ConstArrayAttr::get(
        cir::ArrayType::get(commonElementType, arrayBound),
        mlir::ArrayAttr::get(builder.getContext(), eles));
  }

  SmallVector<mlir::Attribute> eles;
  eles.reserve(elements.size());
  for (auto const &element : elements)
    eles.push_back(element);

  auto arrAttr = mlir::ArrayAttr::get(builder.getContext(), eles);
  return builder.getAnonConstRecord(arrAttr, /*packed=*/true);
}

} // namespace

//===----------------------------------------------------------------------===//
//                          ConstantLValueEmitter
//===----------------------------------------------------------------------===//

namespace {
/// A struct which can be used to peephole certain kinds of finalization
/// that normally happen during l-value emission.
struct ConstantLValue {
  llvm::PointerUnion<mlir::Value, mlir::Attribute> value;
  bool hasOffsetApplied;

  /*implicit*/ ConstantLValue(std::nullptr_t)
      : value(nullptr), hasOffsetApplied(false) {}
  /*implicit*/ ConstantLValue(cir::GlobalViewAttr address)
      : value(address), hasOffsetApplied(false) {}

  ConstantLValue() : value(nullptr), hasOffsetApplied(false) {}
};

/// A helper class for emitting constant l-values.
class ConstantLValueEmitter
    : public ConstStmtVisitor<ConstantLValueEmitter, ConstantLValue> {
  CIRGenModule &cgm;
  ConstantEmitter &emitter;
  const APValue &value;
  QualType destType;

  // Befriend StmtVisitorBase so that we don't have to expose Visit*.
  friend StmtVisitorBase;

public:
  ConstantLValueEmitter(ConstantEmitter &emitter, const APValue &value,
                        QualType destType)
      : cgm(emitter.cgm), emitter(emitter), value(value), destType(destType) {}

  mlir::Attribute tryEmit();

private:
  mlir::Attribute tryEmitAbsolute(mlir::Type destTy);
  ConstantLValue tryEmitBase(const APValue::LValueBase &base);

  ConstantLValue VisitStmt(const Stmt *s) { return nullptr; }
  ConstantLValue VisitConstantExpr(const ConstantExpr *e);
  ConstantLValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *e);
  ConstantLValue VisitStringLiteral(const StringLiteral *e);
  ConstantLValue VisitObjCBoxedExpr(const ObjCBoxedExpr *e);
  ConstantLValue VisitObjCEncodeExpr(const ObjCEncodeExpr *e);
  ConstantLValue VisitObjCStringLiteral(const ObjCStringLiteral *e);
  ConstantLValue VisitPredefinedExpr(const PredefinedExpr *e);
  ConstantLValue VisitAddrLabelExpr(const AddrLabelExpr *e);
  ConstantLValue VisitCallExpr(const CallExpr *e);
  ConstantLValue VisitBlockExpr(const BlockExpr *e);
  ConstantLValue VisitCXXTypeidExpr(const CXXTypeidExpr *e);
  ConstantLValue
  VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *e);

  /// Return GEP-like value offset
  mlir::ArrayAttr getOffset(mlir::Type ty) {
    int64_t offset = value.getLValueOffset().getQuantity();
    cir::CIRDataLayout layout(cgm.getModule());
    SmallVector<int64_t, 3> idxVec;
    cgm.getBuilder().computeGlobalViewIndicesFromFlatOffset(offset, ty, layout,
                                                            idxVec);

    llvm::SmallVector<mlir::Attribute, 3> indices;
    for (int64_t i : idxVec) {
      mlir::IntegerAttr intAttr = cgm.getBuilder().getI32IntegerAttr(i);
      indices.push_back(intAttr);
    }

    if (indices.empty())
      return {};
    return cgm.getBuilder().getArrayAttr(indices);
  }

  /// Apply the value offset to the given constant.
  ConstantLValue applyOffset(ConstantLValue &c) {
    // Handle attribute constant LValues.
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(c.value)) {
      if (auto gv = mlir::dyn_cast<cir::GlobalViewAttr>(attr)) {
        auto baseTy = mlir::cast<cir::PointerType>(gv.getType()).getPointee();
        mlir::Type destTy = cgm.getTypes().convertTypeForMem(destType);
        assert(!gv.getIndices() && "Global view is already indexed");
        return cir::GlobalViewAttr::get(destTy, gv.getSymbol(),
                                        getOffset(baseTy));
      }
      llvm_unreachable("Unsupported attribute type to offset");
    }

    cgm.errorNYI("ConstantLValue: non-attribute offset");
    return {};
  }
};

} // namespace

mlir::Attribute ConstantLValueEmitter::tryEmit() {
  const APValue::LValueBase &base = value.getLValueBase();

  // The destination type should be a pointer or reference
  // type, but it might also be a cast thereof.
  //
  // FIXME: the chain of casts required should be reflected in the APValue.
  // We need this in order to correctly handle things like a ptrtoint of a
  // non-zero null pointer and addrspace casts that aren't trivially
  // represented in LLVM IR.
  mlir::Type destTy = cgm.getTypes().convertTypeForMem(destType);
  assert(mlir::isa<cir::PointerType>(destTy));

  // If there's no base at all, this is a null or absolute pointer,
  // possibly cast back to an integer type.
  if (!base)
    return tryEmitAbsolute(destTy);

  // Otherwise, try to emit the base.
  ConstantLValue result = tryEmitBase(base);

  // If that failed, we're done.
  llvm::PointerUnion<mlir::Value, mlir::Attribute> &value = result.value;
  if (!value)
    return {};

  // Apply the offset if necessary and not already done.
  if (!result.hasOffsetApplied)
    value = applyOffset(result).value;

  // Convert to the appropriate type; this could be an lvalue for
  // an integer. FIXME: performAddrSpaceCast
  if (mlir::isa<cir::PointerType>(destTy)) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(value))
      return attr;
    cgm.errorNYI("ConstantLValueEmitter: non-attribute pointer");
    return {};
  }

  cgm.errorNYI("ConstantLValueEmitter: other?");
  return {};
}

/// Try to emit an absolute l-value, such as a null pointer or an integer
/// bitcast to pointer type.
mlir::Attribute ConstantLValueEmitter::tryEmitAbsolute(mlir::Type destTy) {
  // If we're producing a pointer, this is easy.
  auto destPtrTy = mlir::cast<cir::PointerType>(destTy);
  return cgm.getBuilder().getConstPtrAttr(
      destPtrTy, value.getLValueOffset().getQuantity());
}

ConstantLValue
ConstantLValueEmitter::tryEmitBase(const APValue::LValueBase &base) {
  // Handle values.
  if (const ValueDecl *d = base.dyn_cast<const ValueDecl *>()) {
    // The constant always points to the canonical declaration. We want to look
    // at properties of the most recent declaration at the point of emission.
    d = cast<ValueDecl>(d->getMostRecentDecl());

    if (d->hasAttr<WeakRefAttr>()) {
      cgm.errorNYI(d->getSourceRange(),
                   "ConstantLValueEmitter: emit pointer base for weakref");
      return {};
    }

    if (auto *fd = dyn_cast<FunctionDecl>(d)) {
      cir::FuncOp fop = cgm.getAddrOfFunction(fd);
      CIRGenBuilderTy &builder = cgm.getBuilder();
      mlir::MLIRContext *mlirContext = builder.getContext();
      return cir::GlobalViewAttr::get(
          builder.getPointerTo(fop.getFunctionType()),
          mlir::FlatSymbolRefAttr::get(mlirContext, fop.getSymNameAttr()));
    }

    if (auto *vd = dyn_cast<VarDecl>(d)) {
      // We can never refer to a variable with local storage.
      if (!vd->hasLocalStorage()) {
        if (vd->isFileVarDecl() || vd->hasExternalStorage())
          return cgm.getAddrOfGlobalVarAttr(vd);

        if (vd->isLocalVarDecl()) {
          cgm.errorNYI(vd->getSourceRange(),
                       "ConstantLValueEmitter: local var decl");
          return {};
        }
      }
    }

    // Classic codegen handles MSGuidDecl,UnnamedGlobalConstantDecl, and
    // TemplateParamObjectDecl, but it can also fall through from VarDecl,
    // in which case it silently returns nullptr. For now, let's emit an
    // error to see what cases we need to handle.
    cgm.errorNYI(d->getSourceRange(),
                 "ConstantLValueEmitter: unhandled value decl");
    return {};
  }

  // Handle typeid(T).
  if (base.dyn_cast<TypeInfoLValue>()) {
    cgm.errorNYI("ConstantLValueEmitter: typeid");
    return {};
  }

  // Otherwise, it must be an expression.
  return Visit(base.get<const Expr *>());
}

ConstantLValue ConstantLValueEmitter::VisitConstantExpr(const ConstantExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: constant expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitCompoundLiteralExpr(const CompoundLiteralExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: compound literal");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitStringLiteral(const StringLiteral *e) {
  return cgm.getAddrOfConstantStringFromLiteral(e);
}

ConstantLValue
ConstantLValueEmitter::VisitObjCEncodeExpr(const ObjCEncodeExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: objc encode expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitObjCStringLiteral(const ObjCStringLiteral *e) {
  cgm.errorNYI(e->getSourceRange(),
               "ConstantLValueEmitter: objc string literal");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitObjCBoxedExpr(const ObjCBoxedExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: objc boxed expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitPredefinedExpr(const PredefinedExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: predefined expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitAddrLabelExpr(const AddrLabelExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: addr label expr");
  return {};
}

ConstantLValue ConstantLValueEmitter::VisitCallExpr(const CallExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: call expr");
  return {};
}

ConstantLValue ConstantLValueEmitter::VisitBlockExpr(const BlockExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: block expr");
  return {};
}

ConstantLValue
ConstantLValueEmitter::VisitCXXTypeidExpr(const CXXTypeidExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: cxx typeid expr");
  return {};
}

ConstantLValue ConstantLValueEmitter::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *e) {
  cgm.errorNYI(e->getSourceRange(),
               "ConstantLValueEmitter: materialize temporary expr");
  return {};
}

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const VarDecl &d) {
  initializeNonAbstract();
  return markIfFailed(tryEmitPrivateForVarInit(d));
}

void ConstantEmitter::finalize(cir::GlobalOp gv) {
  assert(initializedNonAbstract &&
         "finalizing emitter that was used for abstract emission?");
  assert(!finalized && "finalizing emitter multiple times");
  assert(!gv.isDeclaration());
#ifndef NDEBUG
  // Note that we might also be Failed.
  finalized = true;
#endif // NDEBUG
}

mlir::Attribute
ConstantEmitter::tryEmitAbstractForInitializer(const VarDecl &d) {
  AbstractStateRAII state(*this, true);
  return tryEmitPrivateForVarInit(d);
}

ConstantEmitter::~ConstantEmitter() {
  assert((!initializedNonAbstract || finalized || failed) &&
         "not finalized after being initialized for non-abstract emission");
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &d) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!d.hasLocalStorage()) {
    QualType ty = cgm.getASTContext().getBaseElementType(d.getType());
    if (ty->isRecordType()) {
      if (const auto *e = dyn_cast_or_null<CXXConstructExpr>(d.getInit())) {
        const CXXConstructorDecl *cd = e->getConstructor();
        // FIXME: we should probably model this more closely to C++ than
        // just emitting a global with zero init (mimic what we do for trivial
        // assignments and whatnots). Since this is for globals shouldn't
        // be a problem for the near future.
        if (cd->isTrivial() && cd->isDefaultConstructor()) {
          const auto *cxxrd = ty->castAsCXXRecordDecl();
          if (cxxrd->getNumBases() != 0) {
            // There may not be anything additional to do here, but this will
            // force us to pause and test this path when it is supported.
            cgm.errorNYI("tryEmitPrivateForVarInit: cxx record with bases");
            return {};
          }
          if (!cgm.getTypes().isZeroInitializable(cxxrd)) {
            // To handle this case, we really need to go through
            // emitNullConstant, but we need an attribute, not a value
            cgm.errorNYI(
                "tryEmitPrivateForVarInit: non-zero-initializable cxx record");
            return {};
          }
          return cir::ZeroAttr::get(cgm.convertType(d.getType()));
        }
      }
    }
  }
  inConstantContext = d.hasConstantInitialization();

  const Expr *e = d.getInit();
  assert(e && "No initializer to emit");

  QualType destType = d.getType();

  if (!destType->isReferenceType()) {
    QualType nonMemoryDestType = getNonMemoryType(cgm, destType);
    if (mlir::Attribute c = ConstExprEmitter(*this).Visit(const_cast<Expr *>(e),
                                                          nonMemoryDestType))
      return emitForMemory(c, destType);
  }

  // Try to emit the initializer.  Note that this can allow some things that
  // are not allowed by tryEmitPrivateForMemory alone.
  if (APValue *value = d.evaluateValue())
    return tryEmitPrivateForMemory(*value, destType);

  return {};
}

mlir::Attribute ConstantEmitter::tryEmitConstantExpr(const ConstantExpr *ce) {
  if (!ce->hasAPValueResult())
    return {};

  QualType retType = ce->getType();
  if (ce->isGLValue())
    retType = cgm.getASTContext().getLValueReferenceType(retType);

  return emitAbstract(ce->getBeginLoc(), ce->getAPValueResult(), retType);
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForMemory(const Expr *e,
                                                         QualType destType) {
  QualType nonMemoryDestType = getNonMemoryType(cgm, destType);
  mlir::TypedAttr c = tryEmitPrivate(e, nonMemoryDestType);
  if (c) {
    mlir::Attribute attr = emitForMemory(c, destType);
    return mlir::cast<mlir::TypedAttr>(attr);
  }
  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForMemory(const APValue &value,
                                                         QualType destType) {
  QualType nonMemoryDestType = getNonMemoryType(cgm, destType);
  mlir::Attribute c = tryEmitPrivate(value, nonMemoryDestType);
  return (c ? emitForMemory(c, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::emitAbstract(const Expr *e,
                                              QualType destType) {
  AbstractStateRAII state{*this, true};
  mlir::Attribute c = mlir::cast<mlir::Attribute>(tryEmitPrivate(e, destType));
  if (!c)
    cgm.errorNYI(e->getSourceRange(),
                 "emitAbstract failed, emit null constaant");
  return c;
}

mlir::Attribute ConstantEmitter::emitAbstract(SourceLocation loc,
                                              const APValue &value,
                                              QualType destType) {
  AbstractStateRAII state(*this, true);
  mlir::Attribute c = tryEmitPrivate(value, destType);
  if (!c)
    cgm.errorNYI(loc, "emitAbstract failed, emit null constaant");
  return c;
}

mlir::Attribute ConstantEmitter::emitNullForMemory(mlir::Location loc,
                                                   CIRGenModule &cgm,
                                                   QualType t) {
  cir::ConstantOp cstOp =
      cgm.emitNullConstant(t, loc).getDefiningOp<cir::ConstantOp>();
  assert(cstOp && "expected cir.const op");
  return emitForMemory(cgm, cstOp.getValue(), t);
}

mlir::Attribute ConstantEmitter::emitForMemory(mlir::Attribute c,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (destType->getAs<AtomicType>()) {
    cgm.errorNYI("emitForMemory: atomic type");
    return {};
  }

  return c;
}

mlir::Attribute ConstantEmitter::emitForMemory(CIRGenModule &cgm,
                                               mlir::Attribute c,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (destType->getAs<AtomicType>()) {
    cgm.errorNYI("atomic constants");
  }

  return c;
}

mlir::TypedAttr ConstantEmitter::tryEmitPrivate(const Expr *e,
                                                QualType destType) {
  assert(!destType->isVoidType() && "can't emit a void constant");

  if (mlir::Attribute c =
          ConstExprEmitter(*this).Visit(const_cast<Expr *>(e), destType))
    return llvm::dyn_cast<mlir::TypedAttr>(c);

  Expr::EvalResult result;

  bool success = false;

  if (destType->isReferenceType())
    success = e->EvaluateAsLValue(result, cgm.getASTContext());
  else
    success =
        e->EvaluateAsRValue(result, cgm.getASTContext(), inConstantContext);

  if (success && !result.hasSideEffects()) {
    mlir::Attribute c = tryEmitPrivate(result.Val, destType);
    return llvm::dyn_cast<mlir::TypedAttr>(c);
  }

  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitPrivate(const APValue &value,
                                                QualType destType) {
  auto &builder = cgm.getBuilder();
  switch (value.getKind()) {
  case APValue::None:
  case APValue::Indeterminate:
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate none or indeterminate");
    return {};
  case APValue::Int: {
    mlir::Type ty = cgm.convertType(destType);
    if (mlir::isa<cir::BoolType>(ty))
      return builder.getCIRBoolAttr(value.getInt().getZExtValue());
    assert(mlir::isa<cir::IntType>(ty) && "expected integral type");
    return cir::IntAttr::get(ty, value.getInt());
  }
  case APValue::Float: {
    const llvm::APFloat &init = value.getFloat();
    if (&init.getSemantics() == &llvm::APFloat::IEEEhalf() &&
        !cgm.getASTContext().getLangOpts().NativeHalfType &&
        cgm.getASTContext().getTargetInfo().useFP16ConversionIntrinsics()) {
      cgm.errorNYI("ConstExprEmitter::tryEmitPrivate half");
      return {};
    }

    mlir::Type ty = cgm.convertType(destType);
    assert(mlir::isa<cir::FPTypeInterface>(ty) &&
           "expected floating-point type");
    return cir::FPAttr::get(ty, init);
  }
  case APValue::Array: {
    const ArrayType *arrayTy = cgm.getASTContext().getAsArrayType(destType);
    const QualType arrayElementTy = arrayTy->getElementType();
    const unsigned numElements = value.getArraySize();
    const unsigned numInitElts = value.getArrayInitializedElts();

    mlir::Attribute filler;
    if (value.hasArrayFiller()) {
      filler = tryEmitPrivate(value.getArrayFiller(), arrayElementTy);
      if (!filler)
        return {};
    }

    SmallVector<mlir::TypedAttr, 16> elements;
    if (filler && builder.isNullValue(filler))
      elements.reserve(numInitElts + 1);
    else
      elements.reserve(numInitElts);

    mlir::Type commonElementType;
    for (unsigned i = 0; i < numInitElts; ++i) {
      const APValue &arrayElement = value.getArrayInitializedElt(i);
      const mlir::Attribute element =
          tryEmitPrivateForMemory(arrayElement, arrayElementTy);
      if (!element)
        return {};

      const mlir::TypedAttr elementTyped = mlir::cast<mlir::TypedAttr>(element);
      if (i == 0)
        commonElementType = elementTyped.getType();
      else if (elementTyped.getType() != commonElementType) {
        commonElementType = {};
      }

      elements.push_back(elementTyped);
    }

    mlir::TypedAttr typedFiller = llvm::cast_or_null<mlir::TypedAttr>(filler);
    if (filler && !typedFiller)
      cgm.errorNYI("array filler should always be typed");

    mlir::Type desiredType = cgm.convertType(destType);
    return emitArrayConstant(cgm, desiredType, commonElementType, numElements,
                             elements, typedFiller);
  }
  case APValue::Vector: {
    const QualType elementType =
        destType->castAs<VectorType>()->getElementType();
    const unsigned numElements = value.getVectorLength();

    SmallVector<mlir::Attribute, 16> elements;
    elements.reserve(numElements);

    for (unsigned i = 0; i < numElements; ++i) {
      const mlir::Attribute element =
          tryEmitPrivateForMemory(value.getVectorElt(i), elementType);
      if (!element)
        return {};
      elements.push_back(element);
    }

    const auto desiredVecTy =
        mlir::cast<cir::VectorType>(cgm.convertType(destType));

    return cir::ConstVectorAttr::get(
        desiredVecTy,
        mlir::ArrayAttr::get(cgm.getBuilder().getContext(), elements));
  }
  case APValue::MemberPointer: {
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate member pointer");
    return {};
  }
  case APValue::LValue:
    return ConstantLValueEmitter(*this, value, destType).tryEmit();
  case APValue::Struct:
  case APValue::Union:
    return ConstRecordBuilder::buildRecord(*this, value, destType);
  case APValue::ComplexInt:
  case APValue::ComplexFloat: {
    mlir::Type desiredType = cgm.convertType(destType);
    cir::ComplexType complexType =
        mlir::dyn_cast<cir::ComplexType>(desiredType);

    mlir::Type complexElemTy = complexType.getElementType();
    if (isa<cir::IntType>(complexElemTy)) {
      llvm::APSInt real = value.getComplexIntReal();
      llvm::APSInt imag = value.getComplexIntImag();
      return builder.getAttr<cir::ConstComplexAttr>(
          complexType, cir::IntAttr::get(complexElemTy, real),
          cir::IntAttr::get(complexElemTy, imag));
    }

    assert(isa<cir::FPTypeInterface>(complexElemTy) &&
           "expected floating-point type");
    llvm::APFloat real = value.getComplexFloatReal();
    llvm::APFloat imag = value.getComplexFloatImag();
    return builder.getAttr<cir::ConstComplexAttr>(
        complexType, cir::FPAttr::get(complexElemTy, real),
        cir::FPAttr::get(complexElemTy, imag));
  }
  case APValue::FixedPoint:
  case APValue::AddrLabelDiff:
    cgm.errorNYI(
        "ConstExprEmitter::tryEmitPrivate fixed point, addr label diff");
    return {};
  }
  llvm_unreachable("Unknown APValue kind");
}

mlir::Value CIRGenModule::emitNullConstant(QualType t, mlir::Location loc) {
  if (t->getAs<PointerType>()) {
    return builder.getNullPtr(getTypes().convertTypeForMem(t), loc);
  }

  if (getTypes().isZeroInitializable(t))
    return builder.getNullValue(getTypes().convertTypeForMem(t), loc);

  if (getASTContext().getAsConstantArrayType(t)) {
    errorNYI("CIRGenModule::emitNullConstant ConstantArrayType");
  }

  if (t->isRecordType())
    errorNYI("CIRGenModule::emitNullConstant RecordType");

  assert(t->isMemberDataPointerType() &&
         "Should only see pointers to data members here!");

  errorNYI("CIRGenModule::emitNullConstant unsupported type");
  return {};
}
