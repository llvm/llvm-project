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
namespace ConstRecordBuilder {
// A class to manage the list of 'initializers' for building the record
// initialization.  This abstracts out the APValue and the InitListExpr.
class RecordBuilderInitList {
  unsigned initIdx = 0;
  bool isUnion = false;
  std::variant<APValue, const InitListExpr *> value;

  bool holdsExpr() const {
    return std::holds_alternative<const InitListExpr *>(value);
  }

  bool holdsAPValue() const { return std::holds_alternative<APValue>(value); }

  const Expr *getExpr() {
    assert(holdsExpr());
    return std::get<const InitListExpr *>(value)->getInit(initIdx);
  }

  mlir::Location getExprLoc(CIRGenModule &cgm) {
    assert(holdsExpr());
    return cgm.getLoc(std::get<const InitListExpr *>(value)->getBeginLoc());
  }

  const APValue getAPVal() {
    assert(holdsAPValue());
    if (isUnion)
      return std::get<APValue>(value).getUnionValue();
    return std::get<APValue>(value).getStructField(initIdx);
  }

public:
  RecordBuilderInitList(const RecordDecl *rd, APValue val)
      : isUnion(rd->isUnion()), value(val) {}
  RecordBuilderInitList(const RecordDecl *rd, const InitListExpr *ile)
      : isUnion(rd->isUnion()), value(ile) {
    assert(ile);
  }

  bool empty() const {
    if (auto *const *ile = std::get_if<const InitListExpr *>(&value))
      return initIdx >= (*ile)->getNumInits();

    // This branch is likely always true, but we guard against it being 'none'
    // anyway.
    if (isUnion)
      return !std::get<APValue>(value).isUnion();

    return initIdx >= std::get<APValue>(value).getStructNumFields();
  }

  const FieldDecl *getActiveUnionField() const {
    if (holdsExpr())
      return std::get<const InitListExpr *>(value)
          ->getInitializedFieldInUnion();
    return std::get<APValue>(value).getUnionField();
  }

  // Return whether this is a field that should be skipped for one reason or
  // another.
  bool shouldSkip(const FieldDecl *fd) {
    if (fd->isUnnamedBitField())
      return true;

    if (holdsExpr() && isa_and_nonnull<NoInitExpr>(getExpr()))
      return true;

    return false;
  }

  // Advance the iterator on a 'skipped' field.  Note in the case of an
  // init-list this doesn't advance if its an unnamed bitfield, as those aren't
  // represented in the AST.
  void advanceSkip(const FieldDecl *fd) {
    assert(!isUnion);
    if (holdsExpr() && fd->isUnnamedBitField())
      return;
    ++initIdx;
  }
  // Advance the iterator on a 'normal' field, which always just increments the
  // index.
  void advance() {
    assert(!isUnion);
    ++initIdx;
  }

  APValue getBase(unsigned idx) {
    // We could potentially handle this with init-list, but we just skip it
    // because classic codegen does. If we decide to, we'll probably have to do
    // something where we get through the init-list elements to make this work
    // right (sub-init-list?).
    assert(holdsAPValue());

    return std::get<APValue>(value).getStructBase(idx);
  }

  bool hasSideEffects(const ASTContext &ctx) {
    if (holdsExpr() && getExpr()->HasSideEffects(ctx))
      return true;
    // APValue never has side effects.
    return false;
  }

  mlir::Attribute emit(ConstantEmitter &emitter, QualType fieldTy) {
    if (holdsExpr()) {
      const Expr *e = getExpr();
      return e ? emitter.tryEmitPrivateForMemory(e, fieldTy)
               : emitter.emitNullForMemory(getExprLoc(emitter.cgm), fieldTy);
    }
    return emitter.tryEmitPrivateForMemory(getAPVal(), fieldTy);
  }
};

llvm::APInt bitfieldStorageToAPInt(mlir::Attribute attr, unsigned storageSize,
                                   bool isBigEndian) {
  // An empty attribute is just zero of the correct size.
  if (!attr)
    return llvm::APInt(storageSize, 0);
  // An int type is just the value held in the attribute.
  if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(attr))
    return intAttr.getValue();

  // Else we are in the array case, we have to create the big APInt, and fill it
  // up.
  llvm::APInt result(storageSize, 0);
  auto elts = mlir::cast<mlir::ArrayAttr>(
      mlir::cast<cir::ConstArrayAttr>(attr).getElts());

  unsigned numBytes = elts.size();
  for (unsigned i = 0; i != numBytes; ++i) {
    unsigned byteIdx = isBigEndian ? numBytes - 1 - i : i;
    llvm::APInt byte =
        mlir::cast<cir::IntAttr>(elts[i]).getValue().zextOrTrunc(8);
    result.insertBits(byte, byteIdx * 8);
  }
  return result;
}

mlir::Attribute apIntToBitfieldStorage(CIRGenModule &cgm,
                                       mlir::Type storageType,
                                       const llvm::APInt &value,
                                       bool isBigEndian) {
  // If we'ere an int, just return the new value.
  if (mlir::isa<cir::IntTypeInterface>(storageType))
    return cir::IntAttr::get(storageType, value);

  // Array of bytes case, fill up an array.
  CIRGenBuilderTy &builder = cgm.getBuilder();
  auto arrayTy = mlir::cast<cir::ArrayType>(storageType);

  unsigned numBytes = arrayTy.getSize();
  cir::IntType byteTy = builder.getUInt8Ty();
  llvm::SmallVector<mlir::Attribute, 8> bytes(numBytes);

  for (unsigned i = 0; i != numBytes; ++i) {
    unsigned byteIdx = isBigEndian ? numBytes - 1 - i : i;
    bytes[i] = cir::IntAttr::get(byteTy, value.extractBits(8, byteIdx * 8));
  }
  return cir::ConstArrayAttr::get(
      arrayTy, mlir::ArrayAttr::get(builder.getContext(), bytes));
}

// Bitfields are lowered to either an integer type, or a series of bytes, see
// getBitfieldStorageType. Because of this, we have to figure out how to store
// this init value in those. Do this by converting the current value in the
// 'storage' type to an APInt so we can do our masking correctly, then convert
// back.  The int path is trivial (getValue/create a new one with the new
// value).  The array type requires breaking it up into its constituent values.
mlir::Attribute updateBitfieldInit(CIRGenModule &cgm,
                                   mlir::Attribute existingVal,
                                   cir::IntAttr newVal, bool isSigned,
                                   const CIRGenBitFieldInfo &bfInfo) {
  bool isBigEndian = cgm.getDataLayout().isBigEndian();
  llvm::APInt result =
      bitfieldStorageToAPInt(existingVal, bfInfo.storageSize, isBigEndian);

  llvm::APInt curValue = newVal.getValue();
  // Make sure we truncate (or properly extend) the existing value for the
  // number of bits in the bitfield. The AST/Sema doesn't do a good job of
  // making sure this is done.
  if (isSigned)
    curValue = curValue.sextOrTrunc(bfInfo.size);
  else
    curValue = curValue.zextOrTrunc(bfInfo.size);

  // Extend to the full storage size so we can shift/mask.
  curValue = curValue.zext(bfInfo.storageSize);

  // bfInfo.offset is already adjusted for endianness, so no endian-changes need
  // to happen here.
  curValue = curValue.shl(bfInfo.offset);
  llvm::APInt mask(bfInfo.storageSize, 0);
  mask.setBits(bfInfo.offset, bfInfo.offset + bfInfo.size);

  result &= ~mask;
  result |= curValue;

  return apIntToBitfieldStorage(cgm, bfInfo.storageType, result, isBigEndian);
}

mlir::Attribute
setBitfieldInit(CIRGenModule &cgm, const CIRGenRecordLayout &cirLayout,
                CIRGenBuilderTy &builder, const FieldDecl *field,
                mlir::Attribute existingVal, mlir::Attribute newVal) {
  const CIRGenBitFieldInfo &info = cirLayout.getBitFieldInfo(field);
  auto intAttr = mlir::dyn_cast<cir::IntAttr>(newVal);
  // This could alternatively be a 'bool' attr here, so do a quick fixup to
  // get the value correctly initialized.
  if (!intAttr) {
    auto boolAttr = mlir::cast<cir::BoolAttr>(newVal);
    intAttr = cir::IntAttr::get(
        builder.getUIntNTy(1), llvm::APInt(/*numBits=*/1, boolAttr.getValue()));
  }

  return updateBitfieldInit(
      cgm, existingVal, intAttr,
      field->getType()->isSignedIntegerOrEnumerationType(), info);
}

mlir::Attribute buildRecordHelper(ConstantEmitter &emitter,
                                  const RecordDecl *rd,
                                  const RecordDecl *vtableBaseTy,
                                  RecordBuilderInitList inits, bool handleBases,
                                  CharUnits offsetInDerived,
                                  bool asBaseSubObj) {
  CIRGenModule &cgm = emitter.cgm;
  CIRGenBuilderTy &builder = cgm.getBuilder();
  const CIRGenRecordLayout &cirLayout =
      cgm.getTypes().getCIRGenRecordLayout(rd);
  cir::RecordType recordTy = asBaseSubObj ? cirLayout.getBaseSubobjectCIRType()
                                          : cirLayout.getCIRType();
  // Unions in CIR are represented by all of their types, so we should be able
  // to just initialize it with whatever the active field is.
  if (rd->isUnion()) {
    if (inits.empty())
      return builder.getZeroInitAttr(recordTy);

    const FieldDecl *activeField = inits.getActiveUnionField();
    if (!activeField || activeField->isZeroSize(cgm.getASTContext()))
      return builder.getZeroInitAttr(recordTy);

    mlir::Attribute eltAttr = inits.emit(emitter, activeField->getType());
    if (!eltAttr)
      return {};

    if (activeField->isBitField())
      eltAttr = setBitfieldInit(cgm, cirLayout, builder, activeField,
                                /*existingVal=*/{}, eltAttr);

    return cir::ConstRecordAttr::get(recordTy, builder.getArrayAttr({eltAttr}));
  }

  llvm::SmallVector<mlir::Attribute> elements(recordTy.getNumElements());

  if (auto *cxxrd = dyn_cast<CXXRecordDecl>(rd)) {
    const ASTRecordLayout &astLayout =
        emitter.cgm.getASTContext().getASTRecordLayout(cxxrd);
    if (astLayout.hasOwnVFPtr()) {
      mlir::Value addrPtr = emitter.cgm.getCXXABI().getVTableAddressPoint(
          BaseSubobject(cxxrd, offsetInDerived),
          cast<CXXRecordDecl>(vtableBaseTy));
      assert(!cir::MissingFeatures::addressPointerAuthInfo());
      auto apOp = addrPtr.getDefiningOp<cir::VTableAddrPointOp>();
      mlir::ArrayAttr indices = builder.getArrayAttr(
          {builder.getI32IntegerAttr(apOp.getAddressPoint().getIndex()),
           builder.getI32IntegerAttr(apOp.getAddressPoint().getOffset())});
      elements[0] =
          cir::GlobalViewAttr::get(cir::VPtrType::get(builder.getContext()),
                                   apOp.getNameAttr(), indices);
    }

    for (auto [idx, base] : llvm::enumerate(cxxrd->bases())) {
      // Our init-list implementation here just skips bases because classic
      // compiler does (see the comment in buildRecord). We perhaps COULD do
      // this, but for now we'll skip them.
      if (!handleBases)
        return {};

      if (base.isVirtual())
        continue;

      const auto *baseDecl = base.getType()->castAsCXXRecordDecl();

      if (!cirLayout.hasNonVirtualBaseCIRField(baseDecl))
        continue;

      APValue baseValue = inits.getBase(idx);

      const ASTRecordLayout &derivedLayout =
          cgm.getASTContext().getASTRecordLayout(cxxrd);
      CharUnits baseOff =
          offsetInDerived + derivedLayout.getBaseClassOffset(baseDecl);

      unsigned baseFieldIdx = cirLayout.getNonVirtualBaseCIRFieldNo(baseDecl);
      elements[baseFieldIdx] = buildRecordHelper(
          emitter, baseDecl, vtableBaseTy, RecordBuilderInitList(rd, baseValue),
          handleBases, baseOff, /*asBaseSubObj=*/true);
    }

    if (cxxrd->getNumVBases()) {
      cgm.errorNYI(cxxrd->getSourceRange(),
                   "buildRecordHelper: virtual base classes");
      return {};
    }
  }

  for (const FieldDecl *field : rd->fields()) {
    // If we don't have any initializers left, we'll just zero-init below. This
    // isn't perfectly accurate to classic compiler, since we are potentially
    // zero-initing padding (instead of leaving it undef), but that is a
    // complexity we can deal with later if we find it necessary.
    if (inits.empty())
      break;

    if (inits.shouldSkip(field)) {
      inits.advanceSkip(field);
      continue;
    }

    // If we didn't lay it out, there is nothing to initialize.  This is
    // either zero size or nothing at all.  IF our init has side effects, we
    // cannot const init this.
    if (!cirLayout.hasCIRField(field)) {
      if (inits.hasSideEffects(cgm.getASTContext()))
        return {};
      inits.advance();
      continue;
    }

    unsigned fieldIdx = cirLayout.getCIRFieldNo(field);

    mlir::Attribute eltAttr = inits.emit(emitter, field->getType());
    inits.advance();

    if (!eltAttr)
      return {};

    if (field->isBitField())
      elements[fieldIdx] = setBitfieldInit(cgm, cirLayout, builder, field,
                                           elements[fieldIdx], eltAttr);
    else
      elements[fieldIdx] = eltAttr;
  }

  // Anything we haven't initialized, we try to zero init. We could/should
  // probably leave the padding as undef if !CGM.ZeroInitPadding, but that ends
  // up being quite an additional bit of complexity (but could be implemented in
  // the field searching above).
  for (unsigned i = 0; i < elements.size(); ++i) {
    if (!elements[i]) {
      elements[i] = builder.getZeroInitAttr(recordTy.getElementType(i));
      if (!elements[i])
        return {};
    }
  }

  return builder.getConstRecordOrZeroAttr(builder.getArrayAttr(elements),
                                          /*packed=*/recordTy.getPacked(),
                                          /*padded=*/recordTy.getPadded(),
                                          recordTy);
}

mlir::Attribute buildRecord(ConstantEmitter &emitter, InitListExpr *ile,
                            QualType valTy) {
  // Bail out if we have base classes. We could support these, but they only
  // arise in C++1z where we will have already constant folded most
  // interesting cases. FIXME: There are still a few more cases we can handle
  // this way.
  const bool handleBases = false;
  const RecordDecl *rd = ile->getType()->castAsRecordDecl();
  return buildRecordHelper(emitter, rd, rd, RecordBuilderInitList(rd, ile),
                           handleBases, CharUnits::Zero(),
                           /*asBaseSubObj=*/false);
}

mlir::Attribute buildRecord(ConstantEmitter &emitter, const APValue &val,
                            QualType valTy) {
  const RecordDecl *rd =
      valTy->castAs<clang::RecordType>()->getDecl()->getDefinitionOrSelf();
  return buildRecordHelper(emitter, rd, rd, RecordBuilderInitList(rd, val),
                           /*handleBases=*/true, CharUnits::Zero(),
                           /*asBaseSubObj=*/false);
}
} // namespace ConstRecordBuilder

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
  [[maybe_unused]] ConstantEmitter &emitter;

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
    if (const auto *ece = dyn_cast<ExplicitCastExpr>(e))
      cgm.emitExplicitCastExprType(ece,
                                   const_cast<CIRGenFunction *>(emitter.cgf));

    Expr *subExpr = e->getSubExpr();

    switch (e->getCastKind()) {
    case CK_ToUnion:
    case CK_AddressSpaceConversion:
    case CK_ReinterpretMemberPointer:
      cgm.errorNYI(e->getBeginLoc(), "ConstExprEmitter::VisitCastExpr");
      return {};

    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer:
      // Return {} to let the APValue evaluator handle member pointer type
      // conversions.  The APValue::MemberPointer case in tryEmitPrivate
      // already builds the correct GEP path for cross-class member pointers.
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
    case CK_HLSLMatrixTruncation:
    case CK_HLSLElementwiseCast:
    case CK_HLSLAggregateSplatCast:
      return {};
    }
    llvm_unreachable("Invalid CastKind");
  }

  mlir::Attribute VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die, QualType t) {
    // No need for a DefaultInitExprScope: we don't handle 'this' in a
    // constant expression.
    return Visit(die->getExpr(), t);
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
    return cgm.getBuilder().getZeroInitAttr(cgm.convertType(t));
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
    if (!e->getConstructor()->isTrivial())
      return {};

    // Only default and copy/move constructors can be trivial.
    if (e->getNumArgs()) {
      assert(e->getNumArgs() == 1 && "trivial ctor with > 1 argument");
      assert(e->getConstructor()->isCopyOrMoveConstructor() &&
             "trivial ctor has argument but isn't a copy/move ctor");

      Expr *arg = e->getArg(0);
      assert(cgm.getASTContext().hasSameUnqualifiedType(ty, arg->getType()) &&
             "argument to copy ctor is of wrong type");

      // Look through the temporary; it's just converting the value to an lvalue
      // to pass it to the constructor.
      if (auto const *mte = dyn_cast<MaterializeTemporaryExpr>(arg))
        return Visit(mte->getSubExpr(), ty);

      // TODO: Investigate whether there are cases that can fall through to here
      //       that need to be handled. This is missing in classic codegen also.
      assert(!cir::MissingFeatures::ctorConstLvalueToRvalueConversion());

      // Don't try to support arbitrary lvalue-to-rvalue conversions for now.
      return {};
    }

    return cgm.getBuilder().getZeroInitAttr(cgm.convertType(ty));
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
  /*implicit*/ ConstantLValue(cir::BlockAddrInfoAttr address)
      : value(address), hasOffsetApplied(true) {}

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
      // Use the destination pointer type (e.g. struct field type), not
      // fop.getFunctionType(), so initializers stay valid when a no-prototype
      // FuncOp is later replaced by a prototyped definition with the same
      // symbol. CIR allows the view type to differ from the symbol's type.
      mlir::Type ptrTy = cgm.getTypes().convertTypeForMem(destType);
      assert(mlir::isa<cir::PointerType>(ptrTy) &&
             "function address in constant must be a pointer");
      return cir::GlobalViewAttr::get(
          ptrTy,
          mlir::FlatSymbolRefAttr::get(mlirContext, fop.getSymNameAttr()));
    }

    if (auto *vd = dyn_cast<VarDecl>(d)) {
      // We can never refer to a variable with local storage.
      if (!vd->hasLocalStorage()) {
        if (vd->isFileVarDecl() || vd->hasExternalStorage())
          return cgm.getAddrOfGlobalVarAttr(vd);

        if (vd->isLocalVarDecl()) {
          cir::GlobalLinkageKind linkage = cgm.getCIRLinkageVarDefinition(vd);
          return cgm.getBuilder().getGlobalViewAttr(
              cgm.getOrCreateStaticVarDecl(*vd, linkage));
        }
      }
    }

    if (isa<MSGuidDecl>(d))
      cgm.errorNYI(d->getSourceRange(), "ConstantLValueEmitter: MSGuidDecl");

    if (const auto *gcd = dyn_cast<UnnamedGlobalConstantDecl>(d))
      return cgm.getBuilder().getGlobalViewAttr(
          cgm.getAddrOfUnnamedGlobalConstantDecl(gcd));

    if (const auto *tpo = dyn_cast<TemplateParamObjectDecl>(d))
      return cgm.getBuilder().getGlobalViewAttr(
          cgm.getAddrOfTemplateParamObject(tpo));

    return {};
  }

  // Handle typeid(T).
  if (TypeInfoLValue typeInfo = base.dyn_cast<TypeInfoLValue>())
    return cast<cir::GlobalViewAttr>(cgm.getAddrOfRTTIDescriptor(
        cgm.getBuilder().getUnknownLoc(), QualType(typeInfo.getType(), 0)));

  // Otherwise, it must be an expression.
  return Visit(base.get<const Expr *>());
}

ConstantLValue ConstantLValueEmitter::VisitConstantExpr(const ConstantExpr *e) {
  cgm.errorNYI(e->getSourceRange(), "ConstantLValueEmitter: constant expr");
  return {};
}

static cir::GlobalViewAttr
tryEmitGlobalCompoundLiteral(ConstantEmitter &emitter,
                             const CompoundLiteralExpr *e) {
  CIRGenModule &cgm = emitter.cgm;
  CIRGenBuilderTy &builder = cgm.getBuilder();
  CharUnits align = cgm.getASTContext().getTypeAlignInChars(e->getType());

  if (cir::GlobalOp addr = cgm.getAddrOfConstantCompoundLiteralIfEmitted(e))
    return builder.getGlobalViewAttr(addr);

  assert(!cir::MissingFeatures::addressSpace());
  mlir::Attribute c =
      emitter.tryEmitForInitializer(e->getInitializer(), e->getType());
  if (!c) {
    assert(!e->isFileScope() &&
           "file-scope compound literal did not have constant initializer!");
    return {};
  }

  auto typedInit = mlir::cast<mlir::TypedAttr>(c);
  bool isConstant = e->getType().isConstantStorage(cgm.getASTContext(),
                                                   /*ExcludeCtor=*/true,
                                                   /*ExcludeDtor=*/false);

  std::string name = cgm.getUniqueGlobalName(".compoundliteral");
  mlir::Location loc = cgm.getLoc(e->getSourceRange());
  cir::GlobalOp gv =
      cgm.createGlobalOp(loc, name, typedInit.getType(), isConstant);
  gv.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
  gv.setAlignment(align.getAsAlign().value());
  CIRGenModule::setInitializer(gv, c);

  emitter.finalize(gv);
  cgm.setAddrOfConstantCompoundLiteral(e, gv);
  return builder.getGlobalViewAttr(gv);
}

ConstantLValue
ConstantLValueEmitter::VisitCompoundLiteralExpr(const CompoundLiteralExpr *e) {
  ConstantEmitter compoundLiteralEmitter(cgm, emitter.cgf);
  compoundLiteralEmitter.setInConstantContext(emitter.isInConstantContext());
  return tryEmitGlobalCompoundLiteral(compoundLiteralEmitter, e);
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
  return cgm.getAddrOfConstantStringFromLiteral(e->getFunctionName());
}

ConstantLValue
ConstantLValueEmitter::VisitAddrLabelExpr(const AddrLabelExpr *e) {
  // A label address taken in a constant context, e.g. a static computed-goto
  // dispatch table `static const void *tbl[] = {&&L1, &&L2}`.  Besides emitting
  // the constant, register the label as address-taken so a following
  // `goto *tbl[i]` lists it among the indirect branch's successors.  A label is
  // always function-local, so cgf is set here.
  assert(emitter.cgf && "label address in a constant requires a function");
  CIRGenFunction &cgf = *const_cast<CIRGenFunction *>(emitter.cgf);
  auto func = cast<cir::FuncOp>(cgf.curFn);
  cir::BlockAddrInfoAttr info = cir::BlockAddrInfoAttr::get(
      &cgf.getMLIRContext(), func.getSymName(), e->getLabel()->getName());
  cgf.indirectGotoTargets.push_back(info);
  return info;
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
  if (e->isTypeOperand())
    return cast<cir::GlobalViewAttr>(
        cgm.getAddrOfRTTIDescriptor(cgm.getLoc(e->getSourceRange()),
                                    e->getTypeOperand(cgm.getASTContext())));
  return cast<cir::GlobalViewAttr>(cgm.getAddrOfRTTIDescriptor(
      cgm.getLoc(e->getSourceRange()), e->getExprOperand()->getType()));
}

ConstantLValue ConstantLValueEmitter::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *e) {
  assert(e->getStorageDuration() == SD_Static);
  const Expr *inner = e->getSubExpr()->skipRValueSubobjectAdjustments();
  mlir::Operation *global = cgm.getAddrOfGlobalTemporary(e, inner);
  return ConstantLValue(
      cgm.getBuilder().getGlobalViewAttr(mlir::cast<cir::GlobalOp>(global)));
}

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const VarDecl &d) {
  initializeNonAbstract();
  return markIfFailed(tryEmitPrivateForVarInit(d));
}

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const Expr *e,
                                                       QualType destType) {
  initializeNonAbstract();
  return markIfFailed(tryEmitPrivateForMemory(e, destType));
}

mlir::Attribute ConstantEmitter::emitForInitializer(const APValue &value,
                                                    QualType destType) {
  initializeNonAbstract();
  auto c = tryEmitPrivateForMemory(value, destType);
  assert(c && "couldn't emit constant value non-abstractly?");
  return c;
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

static mlir::TypedAttr emitNullConstantForBase(CIRGenModule &cgm,
                                               mlir::Type baseType,
                                               const CXXRecordDecl *baseDecl);

static mlir::TypedAttr emitNullConstant(CIRGenModule &cgm, const RecordDecl *rd,
                                        bool asCompleteObject) {
  const CIRGenRecordLayout &layout = cgm.getTypes().getCIRGenRecordLayout(rd);
  mlir::Type ty = (asCompleteObject ? layout.getCIRType()
                                    : layout.getBaseSubobjectCIRType());
  auto recordTy = mlir::cast<cir::RecordType>(ty);

  unsigned numElements = rd->isUnion() ? 1 : recordTy.getNumElements();
  SmallVector<mlir::Attribute> elements(numElements);

  auto *cxxrd = dyn_cast<CXXRecordDecl>(rd);
  // Fill in all the bases.
  if (cxxrd) {
    for (const CXXBaseSpecifier &base : cxxrd->bases()) {
      if (base.isVirtual()) {
        // Ignore virtual bases; if we're laying out for a complete
        // object, we'll lay these out later.
        continue;
      }

      const auto *baseDecl = base.getType()->castAsCXXRecordDecl();
      // Ignore empty bases.
      if (isEmptyRecordForLayout(cgm.getASTContext(), base.getType()) ||
          cgm.getASTContext()
              .getASTRecordLayout(baseDecl)
              .getNonVirtualSize()
              .isZero())
        continue;

      unsigned fieldIndex = layout.getNonVirtualBaseCIRFieldNo(baseDecl);
      mlir::Type baseType = recordTy.getElementType(fieldIndex);
      elements[fieldIndex] = emitNullConstantForBase(cgm, baseType, baseDecl);
    }
  }

  // Fill in all the fields.
  for (const FieldDecl *field : rd->fields()) {
    // Fill in non-bitfields. (Bitfields always use a zero pattern, which we
    // will fill in later.)
    if (!field->isBitField() &&
        !isEmptyFieldForLayout(cgm.getASTContext(), field)) {
      unsigned fieldIndex = layout.getCIRFieldNo(field);
      elements[fieldIndex] = cgm.emitNullConstantAttr(field->getType());
    }

    // For unions, stop after the first named field.
    if (rd->isUnion()) {
      if (field->getIdentifier())
        break;
      if (const auto *fieldRD = field->getType()->getAsRecordDecl())
        if (fieldRD->findFirstNamedDataMember())
          break;
    }
  }

  // Fill in the virtual bases, if we're working with the complete object.
  if (cxxrd && asCompleteObject) {
    for ([[maybe_unused]] const CXXBaseSpecifier &vbase : cxxrd->vbases()) {
      cgm.errorNYI(vbase.getSourceRange(), "emitNullConstant: virtual base");
      return {};
    }
  }

  // Now go through all other fields and zero them out.
  for (unsigned i = 0; i != numElements; ++i) {
    if (!elements[i])
      elements[i] =
          cgm.getBuilder().getZeroInitAttr(recordTy.getElementType(i));
  }

  mlir::MLIRContext *mlirContext = recordTy.getContext();
  return cir::ConstRecordAttr::get(recordTy,
                                   mlir::ArrayAttr::get(mlirContext, elements));
}

/// Emit the null constant for a base subobject.
static mlir::TypedAttr emitNullConstantForBase(CIRGenModule &cgm,
                                               mlir::Type baseType,
                                               const CXXRecordDecl *baseDecl) {
  const CIRGenRecordLayout &baseLayout =
      cgm.getTypes().getCIRGenRecordLayout(baseDecl);

  // Just zero out bases that don't have any pointer to data members.
  if (baseLayout.isZeroInitializableAsBase())
    return cgm.getBuilder().getZeroInitAttr(baseType);

  // Otherwise, we can just use its null constant.
  return emitNullConstant(cgm, baseDecl, /*asCompleteObject=*/false);
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
        if (cd->isTrivial() && cd->isDefaultConstructor())
          return cgm.emitNullConstantAttr(d.getType());
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
  if (const APValue *value = d.evaluateValue())
    return tryEmitPrivateForMemory(*value, destType);

  return {};
}

mlir::Attribute ConstantEmitter::tryEmitAbstract(const Expr *e,
                                                 QualType destType) {
  AbstractStateRAII state{*this, true};
  return tryEmitPrivate(e, destType);
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
  return emitForMemory(cgm, c, destType);
}

mlir::Attribute ConstantEmitter::emitForMemory(CIRGenModule &cgm,
                                               mlir::Attribute c,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (const auto *at = destType->getAs<AtomicType>()) {
    QualType destValueType = at->getValueType();
    c = emitForMemory(cgm, c, destValueType);

    uint64_t innerSize = cgm.getASTContext().getTypeSize(destValueType);
    uint64_t outerSize = cgm.getASTContext().getTypeSize(destType);
    if (innerSize == outerSize)
      return c;

    assert(innerSize < outerSize && "emitted over-large constant for atomic");
    cgm.errorNYI("emitForMemory: tail padding in atomic initializer");
  }

  // In HLSL bool vectors are stored in memory as a vector of i32
  if (destType->isExtVectorBoolType() &&
      !destType->isPackedVectorBoolType(cgm.getASTContext())) {
    cgm.errorNYI("emitForMemory: zero-extend HLSL bool vectors");
  }

  if (destType->isBitIntType()) {
    cgm.errorNYI("emitForMemory: _BitInt type");
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
    mlir::Type ty = cgm.convertType(destType);
    assert(mlir::isa<cir::FPTypeInterface>(ty) &&
           "expected floating-point type");
    return cir::FPAttr::get(ty, value.getFloat());
  }
  case APValue::Array: {
    const ArrayType *arrayTy = cgm.getASTContext().getAsArrayType(destType);
    const QualType arrayElementTy = arrayTy->getElementType();
    const unsigned numElements = value.getArraySize();
    const unsigned numInitElts = value.getArrayInitializedElts();

    mlir::TypedAttr filler;
    if (value.hasArrayFiller()) {
      mlir::Attribute fillerTemp =
          tryEmitPrivate(value.getArrayFiller(), arrayElementTy);
      if (!fillerTemp)
        return {};
      filler = dyn_cast<mlir::TypedAttr>(fillerTemp);
      if (!filler) {
        cgm.errorNYI("ConstExprEmitter::tryEmitPrivate array filler should "
                     "always be typed");
        return {};
      }
    }

    CIRGenBuilderTy &builder = cgm.getBuilder();
    cir::ArrayType desiredType =
        cast<cir::ArrayType>(cgm.convertType(destType));

    llvm::SmallVector<mlir::Attribute> elts;
    if (!filler || builder.isNullValue(filler))
      elts.reserve(numInitElts);
    else
      elts.reserve(numElements);

    // Fill in the known values.
    for (unsigned i = 0; i < numInitElts; ++i) {
      const APValue &arrayElement = value.getArrayInitializedElt(i);
      const mlir::Attribute element =
          tryEmitPrivateForMemory(arrayElement, arrayElementTy);
      if (!element)
        return {};

      elts.push_back(element);
    }

    // If we have an actual value we have to insert for the filler, do so now.
    if (filler && !builder.isNullValue(filler))
      elts.insert(elts.end(), numElements - elts.size(), filler);

    // Remove all null values at the end, so they become 'trailing zeroes'.
    while (!elts.empty() && builder.isNullValue(elts.back()))
      elts.pop_back();

    // For flexible array members, we need to adjust the size of our result to
    // match this.
    if (desiredType.getSize() == 0 && numElements > 0) {
      desiredType =
          cir::ArrayType::get(desiredType.getElementType(), numElements);
    }

    if (elts.empty())
      return cir::ZeroAttr::get(desiredType);

    return cir::ConstArrayAttr::get(
        desiredType, mlir::ArrayAttr::get(builder.getContext(), elts));
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
    assert(!cir::MissingFeatures::cxxABI());

    const ValueDecl *memberDecl = value.getMemberPointerDecl();
    if (!memberDecl)
      return builder.getZeroInitAttr(cgm.convertType(destType));

    if (auto const *cxxDecl = dyn_cast<CXXMethodDecl>(memberDecl)) {
      auto ty = mlir::cast<cir::MethodType>(cgm.convertType(destType));
      if (cxxDecl->isVirtual())
        return cgm.getCXXABI().buildVirtualMethodAttr(ty, cxxDecl);

      cir::FuncOp methodFuncOp =
          cgm.getAddrOfFunction(cxxDecl, ty.getMemberFuncTy());
      return cgm.getBuilder().getMethodAttr(ty, methodFuncOp);
    }

    auto cirTy = mlir::cast<cir::DataMemberType>(cgm.convertType(destType));
    const auto *fieldDecl = cast<FieldDecl>(memberDecl);
    const auto *mpt = destType->castAs<MemberPointerType>();
    const auto *destClass = mpt->getMostRecentCXXRecordDecl();

    // Empty [[no_unique_address]] fields have no CIR field index; represent the
    // pointer-to-data-member by its concrete byte offset.
    if (cgm.isEmptyFieldForMemberPointer(fieldDecl)) {
      const ASTContext &astContext = cgm.getASTContext();
      CharUnits offset =
          astContext.getMemberPointerPathAdjustment(value) +
          astContext.toCharUnitsFromBits(astContext.getFieldOffset(fieldDecl));
      return cir::DataMemberOffsetAttr::get(cirTy, offset.getQuantity());
    }

    std::optional<llvm::SmallVector<int32_t>> path =
        cgm.buildMemberPath(destClass, fieldDecl);
    if (!path)
      return {};
    return builder.getDataMemberAttr(cirTy, *path);
  }
  case APValue::LValue:
    return ConstantLValueEmitter(*this, value, destType).tryEmit();
  case APValue::Struct:
  case APValue::Union:
    return ConstRecordBuilder::buildRecord(*this, value, destType);
  case APValue::ComplexInt:
  case APValue::ComplexFloat: {
    mlir::Type desiredType = cgm.convertType(destType);
    auto complexType = mlir::dyn_cast<cir::ComplexType>(desiredType);

    mlir::Type complexElemTy = complexType.getElementType();
    if (isa<cir::IntType>(complexElemTy)) {
      const llvm::APSInt &real = value.getComplexIntReal();
      const llvm::APSInt &imag = value.getComplexIntImag();
      return cir::ConstComplexAttr::get(builder.getContext(), complexType,
                                        cir::IntAttr::get(complexElemTy, real),
                                        cir::IntAttr::get(complexElemTy, imag));
    }

    assert(isa<cir::FPTypeInterface>(complexElemTy) &&
           "expected floating-point type");
    const llvm::APFloat &real = value.getComplexFloatReal();
    const llvm::APFloat &imag = value.getComplexFloatImag();
    return cir::ConstComplexAttr::get(builder.getContext(), complexType,
                                      cir::FPAttr::get(complexElemTy, real),
                                      cir::FPAttr::get(complexElemTy, imag));
  }
  case APValue::FixedPoint:
  case APValue::AddrLabelDiff:
    cgm.errorNYI(
        "ConstExprEmitter::tryEmitPrivate fixed point, addr label diff");
    return {};
  case APValue::Matrix:
    cgm.errorNYI("ConstExprEmitter::tryEmitPrivate matrix");
    return {};
  }
  llvm_unreachable("Unknown APValue kind");
}

mlir::Value CIRGenModule::emitNullConstant(QualType t, mlir::Location loc) {
  return builder.getConstant(loc, emitNullConstantAttr(t));
}

mlir::TypedAttr CIRGenModule::emitNullConstantAttr(QualType t) {
  if (t->getAs<PointerType>())
    return builder.getConstNullPtrAttr(getTypes().convertTypeForMem(t));

  if (getTypes().isZeroInitializable(t))
    return builder.getZeroInitAttr(getTypes().convertTypeForMem(t));

  if (getASTContext().getAsConstantArrayType(t)) {
    errorNYI("CIRGenModule::emitNullConstantAttr ConstantArrayType");
    return {};
  }

  if (const RecordType *rt = t->getAs<RecordType>())
    return ::emitNullConstant(*this, rt->getDecl(), /*asCompleteObject=*/true);

  assert(t->isMemberDataPointerType() &&
         "Should only see pointers to data members here!");

  return emitNullMemberAttr(t, t->castAs<MemberPointerType>());
}

mlir::TypedAttr
CIRGenModule::emitNullConstantForBase(const CXXRecordDecl *record) {
  return ::emitNullConstant(*this, record, false);
}
