//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CIRGenConstantEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include <optional>

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

/// Get the address of a zero-sized field within a record. The resulting address
/// doesn't necessarily have the right type.
Address CIRGenFunction::emitAddrOfFieldStorage(Address base,
                                               const FieldDecl *field,
                                               llvm::StringRef fieldName,
                                               unsigned fieldIndex) {
  if (field->isZeroSize(getContext())) {
    cgm.errorNYI(field->getSourceRange(),
                 "emitAddrOfFieldStorage: zero-sized field");
    return Address::invalid();
  }

  mlir::Location loc = getLoc(field->getLocation());

  mlir::Type fieldType = convertType(field->getType());
  auto fieldPtr = cir::PointerType::get(fieldType);
  // For most cases fieldName is the same as field->getName() but for lambdas,
  // which do not currently carry the name, so it can be passed down from the
  // CaptureStmt.
  cir::GetMemberOp memberAddr = builder.createGetMember(
      loc, fieldPtr, base.getPointer(), fieldName, fieldIndex);

  // Retrieve layout information, compute alignment and return the final
  // address.
  const RecordDecl *rec = field->getParent();
  const CIRGenRecordLayout &layout = cgm.getTypes().getCIRGenRecordLayout(rec);
  unsigned idx = layout.getCIRFieldNo(field);
  CharUnits offset = CharUnits::fromQuantity(
      layout.getCIRType().getElementOffset(cgm.getDataLayout().layout, idx));
  return Address(memberAddr, base.getAlignment().alignmentAtOffset(offset));
}

/// Given an expression of pointer type, try to
/// derive a more accurate bound on the alignment of the pointer.
Address CIRGenFunction::emitPointerWithAlignment(const Expr *expr,
                                                 LValueBaseInfo *baseInfo) {
  // We allow this with ObjC object pointers because of fragile ABIs.
  assert(expr->getType()->isPointerType() ||
         expr->getType()->isObjCObjectPointerType());
  expr = expr->IgnoreParens();

  // Casts:
  if (auto const *ce = dyn_cast<CastExpr>(expr)) {
    if (const auto *ece = dyn_cast<ExplicitCastExpr>(ce))
      cgm.emitExplicitCastExprType(ece);

    switch (ce->getCastKind()) {
    // Non-converting casts (but not C's implicit conversion from void*).
    case CK_BitCast:
    case CK_NoOp:
    case CK_AddressSpaceConversion: {
      if (const auto *ptrTy =
              ce->getSubExpr()->getType()->getAs<PointerType>()) {
        if (ptrTy->getPointeeType()->isVoidType())
          break;

        LValueBaseInfo innerBaseInfo;
        assert(!cir::MissingFeatures::opTBAA());
        Address addr =
            emitPointerWithAlignment(ce->getSubExpr(), &innerBaseInfo);
        if (baseInfo)
          *baseInfo = innerBaseInfo;

        if (isa<ExplicitCastExpr>(ce)) {
          LValueBaseInfo targetTypeBaseInfo;

          const QualType pointeeType = expr->getType()->getPointeeType();
          const CharUnits align =
              cgm.getNaturalTypeAlignment(pointeeType, &targetTypeBaseInfo);

          // If the source l-value is opaque, honor the alignment of the
          // casted-to type.
          if (innerBaseInfo.getAlignmentSource() != AlignmentSource::Decl) {
            if (baseInfo)
              baseInfo->mergeForCast(targetTypeBaseInfo);
            addr = Address(addr.getPointer(), addr.getElementType(), align);
          }
        }

        assert(!cir::MissingFeatures::sanitizers());

        const mlir::Type eltTy =
            convertTypeForMem(expr->getType()->getPointeeType());
        addr = getBuilder().createElementBitCast(getLoc(expr->getSourceRange()),
                                                 addr, eltTy);
        assert(!cir::MissingFeatures::addressSpace());

        return addr;
      }
      break;
    }

    // Array-to-pointer decay. TODO(cir): BaseInfo and TBAAInfo.
    case CK_ArrayToPointerDecay:
      return emitArrayToPointerDecay(ce->getSubExpr(), baseInfo);

    case CK_UncheckedDerivedToBase:
    case CK_DerivedToBase: {
      assert(!cir::MissingFeatures::opTBAA());
      assert(!cir::MissingFeatures::addressIsKnownNonNull());
      Address addr = emitPointerWithAlignment(ce->getSubExpr(), baseInfo);
      const CXXRecordDecl *derived =
          ce->getSubExpr()->getType()->getPointeeCXXRecordDecl();
      return getAddressOfBaseClass(addr, derived, ce->path(),
                                   shouldNullCheckClassCastValue(ce),
                                   ce->getExprLoc());
    }

    case CK_AnyPointerToBlockPointerCast:
    case CK_BaseToDerived:
    case CK_BaseToDerivedMemberPointer:
    case CK_BlockPointerToObjCPointerCast:
    case CK_BuiltinFnToFnPtr:
    case CK_CPointerToObjCPointerCast:
    case CK_DerivedToBaseMemberPointer:
    case CK_Dynamic:
    case CK_FunctionToPointerDecay:
    case CK_IntegralToPointer:
    case CK_LValueToRValue:
    case CK_LValueToRValueBitCast:
    case CK_NullToMemberPointer:
    case CK_NullToPointer:
    case CK_ReinterpretMemberPointer:
      // Common pointer conversions, nothing to do here.
      // TODO: Is there any reason to treat base-to-derived conversions
      // specially?
      break;

    case CK_ARCConsumeObject:
    case CK_ARCExtendBlockObject:
    case CK_ARCProduceObject:
    case CK_ARCReclaimReturnedObject:
    case CK_AtomicToNonAtomic:
    case CK_BooleanToSignedIntegral:
    case CK_ConstructorConversion:
    case CK_CopyAndAutoreleaseBlockObject:
    case CK_Dependent:
    case CK_FixedPointCast:
    case CK_FixedPointToBoolean:
    case CK_FixedPointToFloating:
    case CK_FixedPointToIntegral:
    case CK_FloatingCast:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToBoolean:
    case CK_FloatingComplexToIntegralComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingRealToComplex:
    case CK_FloatingToBoolean:
    case CK_FloatingToFixedPoint:
    case CK_FloatingToIntegral:
    case CK_HLSLAggregateSplatCast:
    case CK_HLSLArrayRValue:
    case CK_HLSLElementwiseCast:
    case CK_HLSLVectorTruncation:
    case CK_IntToOCLSampler:
    case CK_IntegralCast:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToBoolean:
    case CK_IntegralComplexToFloatingComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralRealToComplex:
    case CK_IntegralToBoolean:
    case CK_IntegralToFixedPoint:
    case CK_IntegralToFloating:
    case CK_LValueBitCast:
    case CK_MatrixCast:
    case CK_MemberPointerToBoolean:
    case CK_NonAtomicToAtomic:
    case CK_ObjCObjectLValueCast:
    case CK_PointerToBoolean:
    case CK_PointerToIntegral:
    case CK_ToUnion:
    case CK_ToVoid:
    case CK_UserDefinedConversion:
    case CK_VectorSplat:
    case CK_ZeroToOCLOpaqueType:
      llvm_unreachable("unexpected cast for emitPointerWithAlignment");
    }
  }

  // Unary &
  if (const UnaryOperator *uo = dyn_cast<UnaryOperator>(expr)) {
    // TODO(cir): maybe we should use cir.unary for pointers here instead.
    if (uo->getOpcode() == UO_AddrOf) {
      LValue lv = emitLValue(uo->getSubExpr());
      if (baseInfo)
        *baseInfo = lv.getBaseInfo();
      assert(!cir::MissingFeatures::opTBAA());
      return lv.getAddress();
    }
  }

  // std::addressof and variants.
  if (auto const *call = dyn_cast<CallExpr>(expr)) {
    switch (call->getBuiltinCallee()) {
    default:
      break;
    case Builtin::BIaddressof:
    case Builtin::BI__addressof:
    case Builtin::BI__builtin_addressof: {
      cgm.errorNYI(expr->getSourceRange(),
                   "emitPointerWithAlignment: builtin addressof");
      return Address::invalid();
    }
    }
  }

  // Otherwise, use the alignment of the type.
  return makeNaturalAddressForPointer(
      emitScalarExpr(expr), expr->getType()->getPointeeType(), CharUnits(),
      /*forPointeeType=*/true, baseInfo);
}

void CIRGenFunction::emitStoreThroughLValue(RValue src, LValue dst,
                                            bool isInit) {
  if (!dst.isSimple()) {
    if (dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element
      const mlir::Location loc = dst.getVectorPointer().getLoc();
      const mlir::Value vector =
          builder.createLoad(loc, dst.getVectorAddress());
      const mlir::Value newVector = builder.create<cir::VecInsertOp>(
          loc, vector, src.getValue(), dst.getVectorIdx());
      builder.createStore(loc, newVector, dst.getVectorAddress());
      return;
    }

    assert(dst.isBitField() && "Unknown LValue type");
    emitStoreThroughBitfieldLValue(src, dst);
    return;

    cgm.errorNYI(dst.getPointer().getLoc(),
                 "emitStoreThroughLValue: non-simple lvalue");
    return;
  }

  assert(!cir::MissingFeatures::opLoadStoreObjC());

  assert(src.isScalar() && "Can't emit an aggregate store with this method");
  emitStoreOfScalar(src.getValue(), dst, isInit);
}

static LValue emitGlobalVarDeclLValue(CIRGenFunction &cgf, const Expr *e,
                                      const VarDecl *vd) {
  QualType t = e->getType();

  // If it's thread_local, emit a call to its wrapper function instead.
  assert(!cir::MissingFeatures::opGlobalThreadLocal());
  if (vd->getTLSKind() == VarDecl::TLS_Dynamic)
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "emitGlobalVarDeclLValue: thread_local variable");

  // Check if the variable is marked as declare target with link clause in
  // device codegen.
  if (cgf.getLangOpts().OpenMP)
    cgf.cgm.errorNYI(e->getSourceRange(), "emitGlobalVarDeclLValue: OpenMP");

  // Traditional LLVM codegen handles thread local separately, CIR handles
  // as part of getAddrOfGlobalVar.
  mlir::Value v = cgf.cgm.getAddrOfGlobalVar(vd);

  assert(!cir::MissingFeatures::addressSpace());
  mlir::Type realVarTy = cgf.convertTypeForMem(vd->getType());
  cir::PointerType realPtrTy = cgf.getBuilder().getPointerTo(realVarTy);
  if (realPtrTy != v.getType())
    v = cgf.getBuilder().createBitcast(v.getLoc(), v, realPtrTy);

  CharUnits alignment = cgf.getContext().getDeclAlign(vd);
  Address addr(v, realVarTy, alignment);
  LValue lv;
  if (vd->getType()->isReferenceType())
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "emitGlobalVarDeclLValue: reference type");
  else
    lv = cgf.makeAddrLValue(addr, t, AlignmentSource::Decl);
  assert(!cir::MissingFeatures::setObjCGCLValueClass());
  return lv;
}

void CIRGenFunction::emitStoreOfScalar(mlir::Value value, Address addr,
                                       bool isVolatile, QualType ty,
                                       bool isInit, bool isNontemporal) {
  assert(!cir::MissingFeatures::opLoadStoreThreadLocal());

  if (const auto *clangVecTy = ty->getAs<clang::VectorType>()) {
    // Boolean vectors use `iN` as storage type.
    if (clangVecTy->isExtVectorBoolType())
      cgm.errorNYI(addr.getPointer().getLoc(),
                   "emitStoreOfScalar ExtVectorBoolType");

    // Handle vectors of size 3 like size 4 for better performance.
    const mlir::Type elementType = addr.getElementType();
    const auto vecTy = cast<cir::VectorType>(elementType);

    // TODO(CIR): Use `ABIInfo::getOptimalVectorMemoryType` once it upstreamed
    if (vecTy.getSize() == 3 && !getLangOpts().PreserveVec3Type)
      cgm.errorNYI(addr.getPointer().getLoc(),
                   "emitStoreOfScalar Vec3 & PreserveVec3Type disabled");
  }

  value = emitToMemory(value, ty);

  assert(!cir::MissingFeatures::opLoadStoreAtomic());

  // Update the alloca with more info on initialization.
  assert(addr.getPointer() && "expected pointer to exist");
  auto srcAlloca = addr.getDefiningOp<cir::AllocaOp>();
  if (currVarDecl && srcAlloca) {
    const VarDecl *vd = currVarDecl;
    assert(vd && "VarDecl expected");
    if (vd->hasInit())
      srcAlloca.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
  }

  assert(currSrcLoc && "must pass in source location");
  builder.createStore(*currSrcLoc, value, addr /*, isVolatile*/);

  if (isNontemporal) {
    cgm.errorNYI(addr.getPointer().getLoc(), "emitStoreOfScalar nontemporal");
    return;
  }

  assert(!cir::MissingFeatures::opTBAA());
}

// TODO: Replace this with a proper TargetInfo function call.
/// Helper method to check if the underlying ABI is AAPCS
static bool isAAPCS(const TargetInfo &targetInfo) {
  return targetInfo.getABI().starts_with("aapcs");
}

mlir::Value CIRGenFunction::emitStoreThroughBitfieldLValue(RValue src,
                                                           LValue dst) {

  const CIRGenBitFieldInfo &info = dst.getBitFieldInfo();
  mlir::Type resLTy = convertTypeForMem(dst.getType());
  Address ptr = dst.getBitFieldAddress();

  bool useVoaltile = cgm.getCodeGenOpts().AAPCSBitfieldWidth &&
                     dst.isVolatileQualified() &&
                     info.volatileStorageSize != 0 && isAAPCS(cgm.getTarget());

  mlir::Value dstAddr = dst.getAddress().getPointer();

  return builder.createSetBitfield(dstAddr.getLoc(), resLTy, ptr,
                                   ptr.getElementType(), src.getValue(), info,
                                   dst.isVolatileQualified(), useVoaltile);
}

RValue CIRGenFunction::emitLoadOfBitfieldLValue(LValue lv, SourceLocation loc) {
  const CIRGenBitFieldInfo &info = lv.getBitFieldInfo();

  // Get the output type.
  mlir::Type resLTy = convertType(lv.getType());
  Address ptr = lv.getBitFieldAddress();

  bool useVoaltile = lv.isVolatileQualified() && info.volatileOffset != 0 &&
                     isAAPCS(cgm.getTarget());

  mlir::Value field =
      builder.createGetBitfield(getLoc(loc), resLTy, ptr, ptr.getElementType(),
                                info, lv.isVolatile(), useVoaltile);
  assert(!cir::MissingFeatures::opLoadEmitScalarRangeCheck() && "NYI");
  return RValue::get(field);
}

Address CIRGenFunction::getAddrOfBitFieldStorage(LValue base,
                                                 const FieldDecl *field,
                                                 mlir::Type fieldType,
                                                 unsigned index) {
  mlir::Location loc = getLoc(field->getLocation());
  cir::PointerType fieldPtr = cir::PointerType::get(fieldType);
  auto rec = cast<cir::RecordType>(base.getAddress().getElementType());
  cir::GetMemberOp sea = getBuilder().createGetMember(
      loc, fieldPtr, base.getPointer(), field->getName(),
      rec.isUnion() ? field->getFieldIndex() : index);
  CharUnits offset = CharUnits::fromQuantity(
      rec.getElementOffset(cgm.getDataLayout().layout, index));
  return Address(sea, base.getAlignment().alignmentAtOffset(offset));
}

LValue CIRGenFunction::emitLValueForBitField(LValue base,
                                             const FieldDecl *field) {
  LValueBaseInfo baseInfo = base.getBaseInfo();
  const CIRGenRecordLayout &layout =
      cgm.getTypes().getCIRGenRecordLayout(field->getParent());
  const CIRGenBitFieldInfo &info = layout.getBitFieldInfo(field);

  assert(!cir::MissingFeatures::preservedAccessIndexRegion());

  unsigned idx = layout.getCIRFieldNo(field);
  Address addr = getAddrOfBitFieldStorage(base, field, info.storageType, idx);

  mlir::Location loc = getLoc(field->getLocation());
  if (addr.getElementType() != info.storageType)
    addr = builder.createElementBitCast(loc, addr, info.storageType);

  QualType fieldType =
      field->getType().withCVRQualifiers(base.getVRQualifiers());
  // TODO(cir): Support TBAA for bit fields.
  assert(!cir::MissingFeatures::opTBAA());
  LValueBaseInfo fieldBaseInfo(baseInfo.getAlignmentSource());
  return LValue::makeBitfield(addr, info, fieldType, fieldBaseInfo);
}

LValue CIRGenFunction::emitLValueForField(LValue base, const FieldDecl *field) {
  LValueBaseInfo baseInfo = base.getBaseInfo();

  if (field->isBitField())
    return emitLValueForBitField(base, field);

  QualType fieldType = field->getType();
  const RecordDecl *rec = field->getParent();
  AlignmentSource baseAlignSource = baseInfo.getAlignmentSource();
  LValueBaseInfo fieldBaseInfo(getFieldAlignmentSource(baseAlignSource));
  assert(!cir::MissingFeatures::opTBAA());

  Address addr = base.getAddress();
  if (auto *classDecl = dyn_cast<CXXRecordDecl>(rec)) {
    if (cgm.getCodeGenOpts().StrictVTablePointers &&
        classDecl->isDynamicClass()) {
      cgm.errorNYI(field->getSourceRange(),
                   "emitLValueForField: strict vtable for dynamic class");
    }
  }

  unsigned recordCVR = base.getVRQualifiers();

  llvm::StringRef fieldName = field->getName();
  unsigned fieldIndex;
  assert(!cir::MissingFeatures::lambdaFieldToName());

  if (rec->isUnion())
    fieldIndex = field->getFieldIndex();
  else {
    const CIRGenRecordLayout &layout =
        cgm.getTypes().getCIRGenRecordLayout(field->getParent());
    fieldIndex = layout.getCIRFieldNo(field);
  }

  addr = emitAddrOfFieldStorage(addr, field, fieldName, fieldIndex);
  assert(!cir::MissingFeatures::preservedAccessIndexRegion());

  // If this is a reference field, load the reference right now.
  if (fieldType->isReferenceType()) {
    cgm.errorNYI(field->getSourceRange(), "emitLValueForField: reference type");
    return LValue();
  }

  if (field->hasAttr<AnnotateAttr>()) {
    cgm.errorNYI(field->getSourceRange(), "emitLValueForField: AnnotateAttr");
    return LValue();
  }

  LValue lv = makeAddrLValue(addr, fieldType, fieldBaseInfo);
  lv.getQuals().addCVRQualifiers(recordCVR);

  // __weak attribute on a field is ignored.
  if (lv.getQuals().getObjCGCAttr() == Qualifiers::Weak) {
    cgm.errorNYI(field->getSourceRange(),
                 "emitLValueForField: __weak attribute");
    return LValue();
  }

  return lv;
}

LValue CIRGenFunction::emitLValueForFieldInitialization(
    LValue base, const clang::FieldDecl *field, llvm::StringRef fieldName) {
  QualType fieldType = field->getType();

  if (!fieldType->isReferenceType())
    return emitLValueForField(base, field);

  const CIRGenRecordLayout &layout =
      cgm.getTypes().getCIRGenRecordLayout(field->getParent());
  unsigned fieldIndex = layout.getCIRFieldNo(field);

  Address v =
      emitAddrOfFieldStorage(base.getAddress(), field, fieldName, fieldIndex);

  // Make sure that the address is pointing to the right type.
  mlir::Type memTy = convertTypeForMem(fieldType);
  v = builder.createElementBitCast(getLoc(field->getSourceRange()), v, memTy);

  // TODO: Generate TBAA information that describes this access as a structure
  // member access and not just an access to an object of the field's type. This
  // should be similar to what we do in EmitLValueForField().
  LValueBaseInfo baseInfo = base.getBaseInfo();
  AlignmentSource fieldAlignSource = baseInfo.getAlignmentSource();
  LValueBaseInfo fieldBaseInfo(getFieldAlignmentSource(fieldAlignSource));
  assert(!cir::MissingFeatures::opTBAA());
  return makeAddrLValue(v, fieldType, fieldBaseInfo);
}

mlir::Value CIRGenFunction::emitToMemory(mlir::Value value, QualType ty) {
  // Bool has a different representation in memory than in registers,
  // but in ClangIR, it is simply represented as a cir.bool value.
  // This function is here as a placeholder for possible future changes.
  return value;
}

void CIRGenFunction::emitStoreOfScalar(mlir::Value value, LValue lvalue,
                                       bool isInit) {
  if (lvalue.getType()->isConstantMatrixType()) {
    assert(0 && "NYI: emitStoreOfScalar constant matrix type");
    return;
  }

  emitStoreOfScalar(value, lvalue.getAddress(), lvalue.isVolatile(),
                    lvalue.getType(), isInit, /*isNontemporal=*/false);
}

mlir::Value CIRGenFunction::emitLoadOfScalar(LValue lvalue,
                                             SourceLocation loc) {
  assert(!cir::MissingFeatures::opLoadStoreThreadLocal());
  assert(!cir::MissingFeatures::opLoadEmitScalarRangeCheck());
  assert(!cir::MissingFeatures::opLoadBooleanRepresentation());

  Address addr = lvalue.getAddress();
  mlir::Type eltTy = addr.getElementType();

  if (mlir::isa<cir::VoidType>(eltTy))
    cgm.errorNYI(loc, "emitLoadOfScalar: void type");

  mlir::Value loadOp = builder.createLoad(getLoc(loc), addr);

  return loadOp;
}

/// Given an expression that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result as an rvalue,
/// returning the rvalue.
RValue CIRGenFunction::emitLoadOfLValue(LValue lv, SourceLocation loc) {
  assert(!lv.getType()->isFunctionType());
  assert(!(lv.getType()->isConstantMatrixType()) && "not implemented");

  if (lv.isBitField())
    return emitLoadOfBitfieldLValue(lv, loc);

  if (lv.isSimple())
    return RValue::get(emitLoadOfScalar(lv, loc));

  if (lv.isVectorElt()) {
    const mlir::Value load =
        builder.createLoad(getLoc(loc), lv.getVectorAddress());
    return RValue::get(builder.create<cir::VecExtractOp>(getLoc(loc), load,
                                                         lv.getVectorIdx()));
  }

  cgm.errorNYI(loc, "emitLoadOfLValue");
  return RValue::get(nullptr);
}

static cir::FuncOp emitFunctionDeclPointer(CIRGenModule &cgm, GlobalDecl gd) {
  assert(!cir::MissingFeatures::weakRefReference());
  return cgm.getAddrOfFunction(gd);
}

static LValue emitFunctionDeclLValue(CIRGenFunction &cgf, const Expr *e,
                                     GlobalDecl gd) {
  const FunctionDecl *fd = cast<FunctionDecl>(gd.getDecl());
  cir::FuncOp funcOp = emitFunctionDeclPointer(cgf.cgm, gd);
  mlir::Location loc = cgf.getLoc(e->getSourceRange());
  CharUnits align = cgf.getContext().getDeclAlign(fd);

  assert(!cir::MissingFeatures::sanitizers());

  mlir::Type fnTy = funcOp.getFunctionType();
  mlir::Type ptrTy = cir::PointerType::get(fnTy);
  mlir::Value addr = cgf.getBuilder().create<cir::GetGlobalOp>(
      loc, ptrTy, funcOp.getSymName());

  if (funcOp.getFunctionType() != cgf.convertType(fd->getType())) {
    fnTy = cgf.convertType(fd->getType());
    ptrTy = cir::PointerType::get(fnTy);

    addr = cir::CastOp::create(cgf.getBuilder(), addr.getLoc(), ptrTy,
                               cir::CastKind::bitcast, addr);
  }

  return cgf.makeAddrLValue(Address(addr, fnTy, align), e->getType(),
                            AlignmentSource::Decl);
}

LValue CIRGenFunction::emitDeclRefLValue(const DeclRefExpr *e) {
  const NamedDecl *nd = e->getDecl();
  QualType ty = e->getType();

  assert(e->isNonOdrUse() != NOUR_Unevaluated &&
         "should not emit an unevaluated operand");

  if (const auto *vd = dyn_cast<VarDecl>(nd)) {
    // Checks for omitted feature handling
    assert(!cir::MissingFeatures::opAllocaStaticLocal());
    assert(!cir::MissingFeatures::opAllocaNonGC());
    assert(!cir::MissingFeatures::opAllocaImpreciseLifetime());
    assert(!cir::MissingFeatures::opAllocaTLS());
    assert(!cir::MissingFeatures::opAllocaOpenMPThreadPrivate());
    assert(!cir::MissingFeatures::opAllocaEscapeByReference());

    // Check if this is a global variable
    if (vd->hasLinkage() || vd->isStaticDataMember())
      return emitGlobalVarDeclLValue(*this, e, vd);

    Address addr = Address::invalid();

    // The variable should generally be present in the local decl map.
    auto iter = localDeclMap.find(vd);
    if (iter != localDeclMap.end()) {
      addr = iter->second;
    } else {
      // Otherwise, it might be static local we haven't emitted yet for some
      // reason; most likely, because it's in an outer function.
      cgm.errorNYI(e->getSourceRange(), "emitDeclRefLValue: static local");
    }

    // Drill into reference types.
    LValue lv =
        vd->getType()->isReferenceType()
            ? emitLoadOfReferenceLValue(addr, getLoc(e->getSourceRange()),
                                        vd->getType(), AlignmentSource::Decl)
            : makeAddrLValue(addr, ty, AlignmentSource::Decl);

    // Statics are defined as globals, so they are not include in the function's
    // symbol table.
    assert((vd->isStaticLocal() || symbolTable.count(vd)) &&
           "non-static locals should be already mapped");

    return lv;
  }

  if (const auto *bd = dyn_cast<BindingDecl>(nd)) {
    if (e->refersToEnclosingVariableOrCapture()) {
      assert(!cir::MissingFeatures::lambdaCaptures());
      cgm.errorNYI(e->getSourceRange(), "emitDeclRefLValue: lambda captures");
      return LValue();
    }
    return emitLValue(bd->getBinding());
  }

  if (const auto *fd = dyn_cast<FunctionDecl>(nd)) {
    LValue lv = emitFunctionDeclLValue(*this, e, fd);

    // Emit debuginfo for the function declaration if the target wants to.
    if (getContext().getTargetInfo().allowDebugInfoForExternalRef())
      assert(!cir::MissingFeatures::generateDebugInfo());

    return lv;
  }

  cgm.errorNYI(e->getSourceRange(), "emitDeclRefLValue: unhandled decl type");
  return LValue();
}

mlir::Value CIRGenFunction::evaluateExprAsBool(const Expr *e) {
  QualType boolTy = getContext().BoolTy;
  SourceLocation loc = e->getExprLoc();

  assert(!cir::MissingFeatures::pgoUse());
  if (e->getType()->getAs<MemberPointerType>()) {
    cgm.errorNYI(e->getSourceRange(),
                 "evaluateExprAsBool: member pointer type");
    return createDummyValue(getLoc(loc), boolTy);
  }

  assert(!cir::MissingFeatures::cgFPOptionsRAII());
  if (!e->getType()->isAnyComplexType())
    return emitScalarConversion(emitScalarExpr(e), e->getType(), boolTy, loc);

  cgm.errorNYI(e->getSourceRange(), "evaluateExprAsBool: complex type");
  return createDummyValue(getLoc(loc), boolTy);
}

LValue CIRGenFunction::emitUnaryOpLValue(const UnaryOperator *e) {
  UnaryOperatorKind op = e->getOpcode();

  // __extension__ doesn't affect lvalue-ness.
  if (op == UO_Extension)
    return emitLValue(e->getSubExpr());

  switch (op) {
  case UO_Deref: {
    QualType t = e->getSubExpr()->getType()->getPointeeType();
    assert(!t.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    assert(!cir::MissingFeatures::opTBAA());
    LValueBaseInfo baseInfo;
    Address addr = emitPointerWithAlignment(e->getSubExpr(), &baseInfo);

    // Tag 'load' with deref attribute.
    // FIXME: This misses some derefence cases and has problematic interactions
    // with other operators.
    if (auto loadOp = addr.getDefiningOp<cir::LoadOp>())
      loadOp.setIsDerefAttr(mlir::UnitAttr::get(&getMLIRContext()));

    LValue lv = makeAddrLValue(addr, t, baseInfo);
    assert(!cir::MissingFeatures::addressSpace());
    assert(!cir::MissingFeatures::setNonGC());
    return lv;
  }
  case UO_Real:
  case UO_Imag: {
    LValue lv = emitLValue(e->getSubExpr());
    assert(lv.isSimple() && "real/imag on non-ordinary l-value");

    // __real is valid on scalars. This is a faster way of testing that.
    // __imag can only produce an rvalue on scalars.
    if (e->getOpcode() == UO_Real &&
        !mlir::isa<cir::ComplexType>(lv.getAddress().getElementType())) {
      assert(e->getSubExpr()->getType()->isArithmeticType());
      return lv;
    }

    QualType exprTy = getContext().getCanonicalType(e->getSubExpr()->getType());
    QualType elemTy = exprTy->castAs<clang::ComplexType>()->getElementType();
    mlir::Location loc = getLoc(e->getExprLoc());
    Address component =
        e->getOpcode() == UO_Real
            ? builder.createComplexRealPtr(loc, lv.getAddress())
            : builder.createComplexImagPtr(loc, lv.getAddress());
    assert(!cir::MissingFeatures::opTBAA());
    LValue elemLV = makeAddrLValue(component, elemTy);
    elemLV.getQuals().addQualifiers(lv.getQuals());
    return elemLV;
  }
  case UO_PreInc:
  case UO_PreDec: {
    cir::UnaryOpKind kind =
        e->isIncrementOp() ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
    LValue lv = emitLValue(e->getSubExpr());

    assert(e->isPrefix() && "Prefix operator in unexpected state!");

    if (e->getType()->isAnyComplexType()) {
      cgm.errorNYI(e->getSourceRange(), "UnaryOp complex inc/dec");
      lv = LValue();
    } else {
      emitScalarPrePostIncDec(e, lv, kind, /*isPre=*/true);
    }

    return lv;
  }
  case UO_Extension:
    llvm_unreachable("UnaryOperator extension should be handled above!");
  case UO_Plus:
  case UO_Minus:
  case UO_Not:
  case UO_LNot:
  case UO_AddrOf:
  case UO_PostInc:
  case UO_PostDec:
  case UO_Coawait:
    llvm_unreachable("UnaryOperator of non-lvalue kind!");
  }
  llvm_unreachable("Unknown unary operator kind!");
}

/// If the specified expr is a simple decay from an array to pointer,
/// return the array subexpression.
/// FIXME: this could be abstracted into a common AST helper.
static const Expr *getSimpleArrayDecayOperand(const Expr *e) {
  // If this isn't just an array->pointer decay, bail out.
  const auto *castExpr = dyn_cast<CastExpr>(e);
  if (!castExpr || castExpr->getCastKind() != CK_ArrayToPointerDecay)
    return nullptr;

  // If this is a decay from variable width array, bail out.
  const Expr *subExpr = castExpr->getSubExpr();
  if (subExpr->getType()->isVariableArrayType())
    return nullptr;

  return subExpr;
}

static cir::IntAttr getConstantIndexOrNull(mlir::Value idx) {
  // TODO(cir): should we consider using MLIRs IndexType instead of IntegerAttr?
  if (auto constantOp = idx.getDefiningOp<cir::ConstantOp>())
    return constantOp.getValueAttr<cir::IntAttr>();
  return {};
}

static CharUnits getArrayElementAlign(CharUnits arrayAlign, mlir::Value idx,
                                      CharUnits eltSize) {
  // If we have a constant index, we can use the exact offset of the
  // element we're accessing.
  if (const cir::IntAttr constantIdx = getConstantIndexOrNull(idx)) {
    const CharUnits offset = constantIdx.getValue().getZExtValue() * eltSize;
    return arrayAlign.alignmentAtOffset(offset);
  }
  // Otherwise, use the worst-case alignment for any element.
  return arrayAlign.alignmentOfArrayElement(eltSize);
}

static QualType getFixedSizeElementType(const ASTContext &astContext,
                                        const VariableArrayType *vla) {
  QualType eltType;
  do {
    eltType = vla->getElementType();
  } while ((vla = astContext.getAsVariableArrayType(eltType)));
  return eltType;
}

static mlir::Value emitArraySubscriptPtr(CIRGenFunction &cgf,
                                         mlir::Location beginLoc,
                                         mlir::Location endLoc, mlir::Value ptr,
                                         mlir::Type eltTy, mlir::Value idx,
                                         bool shouldDecay) {
  CIRGenModule &cgm = cgf.getCIRGenModule();
  // TODO(cir): LLVM codegen emits in bound gep check here, is there anything
  // that would enhance tracking this later in CIR?
  assert(!cir::MissingFeatures::emitCheckedInBoundsGEP());
  return cgm.getBuilder().getArrayElement(beginLoc, endLoc, ptr, eltTy, idx,
                                          shouldDecay);
}

static Address emitArraySubscriptPtr(CIRGenFunction &cgf,
                                     mlir::Location beginLoc,
                                     mlir::Location endLoc, Address addr,
                                     QualType eltType, mlir::Value idx,
                                     mlir::Location loc, bool shouldDecay) {

  // Determine the element size of the statically-sized base.  This is
  // the thing that the indices are expressed in terms of.
  if (const VariableArrayType *vla =
          cgf.getContext().getAsVariableArrayType(eltType)) {
    eltType = getFixedSizeElementType(cgf.getContext(), vla);
  }

  // We can use that to compute the best alignment of the element.
  const CharUnits eltSize = cgf.getContext().getTypeSizeInChars(eltType);
  const CharUnits eltAlign =
      getArrayElementAlign(addr.getAlignment(), idx, eltSize);

  assert(!cir::MissingFeatures::preservedAccessIndexRegion());
  const mlir::Value eltPtr =
      emitArraySubscriptPtr(cgf, beginLoc, endLoc, addr.getPointer(),
                            addr.getElementType(), idx, shouldDecay);
  const mlir::Type elementType = cgf.convertTypeForMem(eltType);
  return Address(eltPtr, elementType, eltAlign);
}

LValue
CIRGenFunction::emitArraySubscriptExpr(const clang::ArraySubscriptExpr *e) {
  if (isa<ExtVectorElementExpr>(e->getBase())) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitArraySubscriptExpr: ExtVectorElementExpr");
    return LValue::makeAddr(Address::invalid(), e->getType(), LValueBaseInfo());
  }

  if (getContext().getAsVariableArrayType(e->getType())) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitArraySubscriptExpr: VariableArrayType");
    return LValue::makeAddr(Address::invalid(), e->getType(), LValueBaseInfo());
  }

  if (e->getType()->getAs<ObjCObjectType>()) {
    cgm.errorNYI(e->getSourceRange(), "emitArraySubscriptExpr: ObjCObjectType");
    return LValue::makeAddr(Address::invalid(), e->getType(), LValueBaseInfo());
  }

  // The index must always be an integer, which is not an aggregate.  Emit it
  // in lexical order (this complexity is, sadly, required by C++17).
  assert((e->getIdx() == e->getLHS() || e->getIdx() == e->getRHS()) &&
         "index was neither LHS nor RHS");

  auto emitIdxAfterBase = [&](bool promote) -> mlir::Value {
    const mlir::Value idx = emitScalarExpr(e->getIdx());

    // Extend or truncate the index type to 32 or 64-bits.
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(idx.getType());
    if (promote && ptrTy && ptrTy.isPtrTo<cir::IntType>())
      cgm.errorNYI(e->getSourceRange(),
                   "emitArraySubscriptExpr: index type cast");
    return idx;
  };

  // If the base is a vector type, then we are forming a vector element
  // with this subscript.
  if (e->getBase()->getType()->isVectorType() &&
      !isa<ExtVectorElementExpr>(e->getBase())) {
    const mlir::Value idx = emitIdxAfterBase(/*promote=*/false);
    const LValue lhs = emitLValue(e->getBase());
    return LValue::makeVectorElt(lhs.getAddress(), idx, e->getBase()->getType(),
                                 lhs.getBaseInfo());
  }

  const mlir::Value idx = emitIdxAfterBase(/*promote=*/true);
  if (const Expr *array = getSimpleArrayDecayOperand(e->getBase())) {
    LValue arrayLV;
    if (const auto *ase = dyn_cast<ArraySubscriptExpr>(array))
      arrayLV = emitArraySubscriptExpr(ase);
    else
      arrayLV = emitLValue(array);

    // Propagate the alignment from the array itself to the result.
    const Address addr = emitArraySubscriptPtr(
        *this, cgm.getLoc(array->getBeginLoc()), cgm.getLoc(array->getEndLoc()),
        arrayLV.getAddress(), e->getType(), idx, cgm.getLoc(e->getExprLoc()),
        /*shouldDecay=*/true);

    const LValue lv = LValue::makeAddr(addr, e->getType(), LValueBaseInfo());

    if (getLangOpts().ObjC && getLangOpts().getGC() != LangOptions::NonGC) {
      cgm.errorNYI(e->getSourceRange(), "emitArraySubscriptExpr: ObjC with GC");
    }

    return lv;
  }

  // The base must be a pointer; emit it with an estimate of its alignment.
  assert(e->getBase()->getType()->isPointerType() &&
         "The base must be a pointer");

  LValueBaseInfo eltBaseInfo;
  const Address ptrAddr = emitPointerWithAlignment(e->getBase(), &eltBaseInfo);
  // Propagate the alignment from the array itself to the result.
  const Address addxr = emitArraySubscriptPtr(
      *this, cgm.getLoc(e->getBeginLoc()), cgm.getLoc(e->getEndLoc()), ptrAddr,
      e->getType(), idx, cgm.getLoc(e->getExprLoc()),
      /*shouldDecay=*/false);

  const LValue lv = LValue::makeAddr(addxr, e->getType(), eltBaseInfo);

  if (getLangOpts().ObjC && getLangOpts().getGC() != LangOptions::NonGC) {
    cgm.errorNYI(e->getSourceRange(), "emitArraySubscriptExpr: ObjC with GC");
  }

  return lv;
}

LValue CIRGenFunction::emitStringLiteralLValue(const StringLiteral *e) {
  cir::GlobalOp globalOp = cgm.getGlobalForStringLiteral(e);
  assert(globalOp.getAlignment() && "expected alignment for string literal");
  unsigned align = *(globalOp.getAlignment());
  mlir::Value addr =
      builder.createGetGlobal(getLoc(e->getSourceRange()), globalOp);
  return makeAddrLValue(
      Address(addr, globalOp.getSymType(), CharUnits::fromQuantity(align)),
      e->getType(), AlignmentSource::Decl);
}

/// Casts are never lvalues unless that cast is to a reference type. If the cast
/// is to a reference, we can have the usual lvalue result, otherwise if a cast
/// is needed by the code generator in an lvalue context, then it must mean that
/// we need the address of an aggregate in order to access one of its members.
/// This can happen for all the reasons that casts are permitted with aggregate
/// result, including noop aggregate casts, and cast from scalar to union.
LValue CIRGenFunction::emitCastLValue(const CastExpr *e) {
  switch (e->getCastKind()) {
  case CK_ToVoid:
  case CK_BitCast:
  case CK_LValueToRValueBitCast:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToMemberPointer:
  case CK_NullToPointer:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_IntegralCast:
  case CK_BooleanToSignedIntegral:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
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
  case CK_DerivedToBaseMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_IntToOCLSampler:
  case CK_FloatingToFixedPoint:
  case CK_FixedPointToFloating:
  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToIntegral:
  case CK_IntegralToFixedPoint:
  case CK_MatrixCast:
  case CK_HLSLVectorTruncation:
  case CK_HLSLArrayRValue:
  case CK_HLSLElementwiseCast:
  case CK_HLSLAggregateSplatCast:
    llvm_unreachable("unexpected cast lvalue");

  case CK_Dependent:
    llvm_unreachable("dependent cast kind in IR gen!");

  case CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  // These are never l-values; just use the aggregate emission code.
  case CK_NonAtomicToAtomic:
  case CK_AtomicToNonAtomic:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_BaseToDerived:
  case CK_AddressSpaceConversion:
  case CK_ObjCObjectLValueCast:
  case CK_VectorSplat:
  case CK_ConstructorConversion:
  case CK_UserDefinedConversion:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_LValueToRValue: {
    cgm.errorNYI(e->getSourceRange(),
                 std::string("emitCastLValue for unhandled cast kind: ") +
                     e->getCastKindName());

    return {};
  }

  case CK_LValueBitCast: {
    // This must be a reinterpret_cast (or c-style equivalent).
    const auto *ce = cast<ExplicitCastExpr>(e);

    cgm.emitExplicitCastExprType(ce, this);
    LValue LV = emitLValue(e->getSubExpr());
    Address V = LV.getAddress().withElementType(
        builder, convertTypeForMem(ce->getTypeAsWritten()->getPointeeType()));

    return makeAddrLValue(V, e->getType(), LV.getBaseInfo());
  }

  case CK_NoOp: {
    // CK_NoOp can model a qualification conversion, which can remove an array
    // bound and change the IR type.
    LValue lv = emitLValue(e->getSubExpr());
    // Propagate the volatile qualifier to LValue, if exists in e.
    if (e->changesVolatileQualification())
      cgm.errorNYI(e->getSourceRange(),
                   "emitCastLValue: NoOp changes volatile qual");
    if (lv.isSimple()) {
      Address v = lv.getAddress();
      if (v.isValid()) {
        mlir::Type ty = convertTypeForMem(e->getType());
        if (v.getElementType() != ty)
          cgm.errorNYI(e->getSourceRange(),
                       "emitCastLValue: NoOp needs bitcast");
      }
    }
    return lv;
  }

  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    auto *derivedClassDecl = e->getSubExpr()->getType()->castAsCXXRecordDecl();

    LValue lv = emitLValue(e->getSubExpr());
    Address thisAddr = lv.getAddress();

    // Perform the derived-to-base conversion
    Address baseAddr =
        getAddressOfBaseClass(thisAddr, derivedClassDecl, e->path(),
                              /*NullCheckValue=*/false, e->getExprLoc());

    // TODO: Support accesses to members of base classes in TBAA. For now, we
    // conservatively pretend that the complete object is of the base class
    // type.
    assert(!cir::MissingFeatures::opTBAA());
    return makeAddrLValue(baseAddr, e->getType(), lv.getBaseInfo());
  }

  case CK_ZeroToOCLOpaqueType:
    llvm_unreachable("NULL to OpenCL opaque type lvalue cast is not valid");
  }

  llvm_unreachable("Invalid cast kind");
}

static DeclRefExpr *tryToConvertMemberExprToDeclRefExpr(CIRGenFunction &cgf,
                                                        const MemberExpr *me) {
  if (auto *vd = dyn_cast<VarDecl>(me->getMemberDecl())) {
    // Try to emit static variable member expressions as DREs.
    return DeclRefExpr::Create(
        cgf.getContext(), NestedNameSpecifierLoc(), SourceLocation(), vd,
        /*RefersToEnclosingVariableOrCapture=*/false, me->getExprLoc(),
        me->getType(), me->getValueKind(), nullptr, nullptr, me->isNonOdrUse());
  }
  return nullptr;
}

LValue CIRGenFunction::emitMemberExpr(const MemberExpr *e) {
  if (DeclRefExpr *dre = tryToConvertMemberExprToDeclRefExpr(*this, e)) {
    emitIgnoredExpr(e->getBase());
    return emitDeclRefLValue(dre);
  }

  Expr *baseExpr = e->getBase();
  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  LValue baseLV;
  if (e->isArrow()) {
    LValueBaseInfo baseInfo;
    assert(!cir::MissingFeatures::opTBAA());
    Address addr = emitPointerWithAlignment(baseExpr, &baseInfo);
    QualType ptrTy = baseExpr->getType()->getPointeeType();
    assert(!cir::MissingFeatures::typeChecks());
    baseLV = makeAddrLValue(addr, ptrTy, baseInfo);
  } else {
    assert(!cir::MissingFeatures::typeChecks());
    baseLV = emitLValue(baseExpr);
  }

  const NamedDecl *nd = e->getMemberDecl();
  if (auto *field = dyn_cast<FieldDecl>(nd)) {
    LValue lv = emitLValueForField(baseLV, field);
    assert(!cir::MissingFeatures::setObjCGCLValueClass());
    if (getLangOpts().OpenMP) {
      // If the member was explicitly marked as nontemporal, mark it as
      // nontemporal. If the base lvalue is marked as nontemporal, mark access
      // to children as nontemporal too.
      cgm.errorNYI(e->getSourceRange(), "emitMemberExpr: OpenMP");
    }
    return lv;
  }

  if (isa<FunctionDecl>(nd)) {
    cgm.errorNYI(e->getSourceRange(), "emitMemberExpr: FunctionDecl");
    return LValue();
  }

  llvm_unreachable("Unhandled member declaration!");
}

/// Evaluate an expression into a given memory location.
void CIRGenFunction::emitAnyExprToMem(const Expr *e, Address location,
                                      Qualifiers quals, bool isInit) {
  // FIXME: This function should take an LValue as an argument.
  switch (getEvaluationKind(e->getType())) {
  case cir::TEK_Complex: {
    LValue lv = makeAddrLValue(location, e->getType());
    emitComplexExprIntoLValue(e, lv, isInit);
    return;
  }

  case cir::TEK_Aggregate: {
    emitAggExpr(e, AggValueSlot::forAddr(location, quals,
                                         AggValueSlot::IsDestructed_t(isInit),
                                         AggValueSlot::IsAliased_t(!isInit),
                                         AggValueSlot::MayOverlap));
    return;
  }

  case cir::TEK_Scalar: {
    RValue rv = RValue::get(emitScalarExpr(e));
    LValue lv = makeAddrLValue(location, e->getType());
    emitStoreThroughLValue(rv, lv);
    return;
  }
  }

  llvm_unreachable("bad evaluation kind");
}

static Address createReferenceTemporary(CIRGenFunction &cgf,
                                        const MaterializeTemporaryExpr *m,
                                        const Expr *inner) {
  // TODO(cir): cgf.getTargetHooks();
  switch (m->getStorageDuration()) {
  case SD_FullExpression:
  case SD_Automatic: {
    QualType ty = inner->getType();

    assert(!cir::MissingFeatures::mergeAllConstants());

    // The temporary memory should be created in the same scope as the extending
    // declaration of the temporary materialization expression.
    cir::AllocaOp extDeclAlloca;
    if (const ValueDecl *extDecl = m->getExtendingDecl()) {
      auto extDeclAddrIter = cgf.localDeclMap.find(extDecl);
      if (extDeclAddrIter != cgf.localDeclMap.end())
        extDeclAlloca = extDeclAddrIter->second.getDefiningOp<cir::AllocaOp>();
    }
    mlir::OpBuilder::InsertPoint ip;
    if (extDeclAlloca)
      ip = {extDeclAlloca->getBlock(), extDeclAlloca->getIterator()};
    return cgf.createMemTemp(ty, cgf.getLoc(m->getSourceRange()),
                             cgf.getCounterRefTmpAsString(), /*alloca=*/nullptr,
                             ip);
  }
  case SD_Thread:
  case SD_Static: {
    cgf.cgm.errorNYI(
        m->getSourceRange(),
        "createReferenceTemporary: static/thread storage duration");
    return Address::invalid();
  }

  case SD_Dynamic:
    llvm_unreachable("temporary can't have dynamic storage duration");
  }
  llvm_unreachable("unknown storage duration");
}

static void pushTemporaryCleanup(CIRGenFunction &cgf,
                                 const MaterializeTemporaryExpr *m,
                                 const Expr *e, Address referenceTemporary) {
  // Objective-C++ ARC:
  //   If we are binding a reference to a temporary that has ownership, we
  //   need to perform retain/release operations on the temporary.
  //
  // FIXME(ogcg): This should be looking at e, not m.
  if (m->getType().getObjCLifetime()) {
    cgf.cgm.errorNYI(e->getSourceRange(), "pushTemporaryCleanup: ObjCLifetime");
    return;
  }

  const QualType::DestructionKind dk = e->getType().isDestructedType();
  if (dk == QualType::DK_none)
    return;

  switch (m->getStorageDuration()) {
  case SD_Static:
  case SD_Thread: {
    CXXDestructorDecl *referenceTemporaryDtor = nullptr;
    if (const auto *classDecl =
            e->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
        classDecl && !classDecl->hasTrivialDestructor())
      // Get the destructor for the reference temporary.
      referenceTemporaryDtor = classDecl->getDestructor();

    if (!referenceTemporaryDtor)
      return;

    cgf.cgm.errorNYI(e->getSourceRange(), "pushTemporaryCleanup: static/thread "
                                          "storage duration with destructors");
    break;
  }

  case SD_FullExpression:
    cgf.pushDestroy(NormalAndEHCleanup, referenceTemporary, e->getType(),
                    CIRGenFunction::destroyCXXObject);
    break;

  case SD_Automatic:
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "pushTemporaryCleanup: automatic storage duration");
    break;

  case SD_Dynamic:
    llvm_unreachable("temporary cannot have dynamic storage duration");
  }
}

LValue CIRGenFunction::emitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *m) {
  const Expr *e = m->getSubExpr();

  assert((!m->getExtendingDecl() || !isa<VarDecl>(m->getExtendingDecl()) ||
          !cast<VarDecl>(m->getExtendingDecl())->isARCPseudoStrong()) &&
         "Reference should never be pseudo-strong!");

  // FIXME: ideally this would use emitAnyExprToMem, however, we cannot do so
  // as that will cause the lifetime adjustment to be lost for ARC
  auto ownership = m->getType().getObjCLifetime();
  if (ownership != Qualifiers::OCL_None &&
      ownership != Qualifiers::OCL_ExplicitNone) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitMaterializeTemporaryExpr: ObjCLifetime");
    return {};
  }

  SmallVector<const Expr *, 2> commaLHSs;
  SmallVector<SubobjectAdjustment, 2> adjustments;
  e = e->skipRValueSubobjectAdjustments(commaLHSs, adjustments);

  for (const Expr *ignored : commaLHSs)
    emitIgnoredExpr(ignored);

  if (isa<OpaqueValueExpr>(e)) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitMaterializeTemporaryExpr: OpaqueValueExpr");
    return {};
  }

  // Create and initialize the reference temporary.
  Address object = createReferenceTemporary(*this, m, e);

  if (auto var = object.getPointer().getDefiningOp<cir::GlobalOp>()) {
    // TODO(cir): add something akin to stripPointerCasts() to ptr above
    cgm.errorNYI(e->getSourceRange(), "emitMaterializeTemporaryExpr: GlobalOp");
    return {};
  } else {
    assert(!cir::MissingFeatures::emitLifetimeMarkers());
    emitAnyExprToMem(e, object, Qualifiers(), /*isInitializer=*/true);
  }
  pushTemporaryCleanup(*this, m, e, object);

  // Perform derived-to-base casts and/or field accesses, to get from the
  // temporary object we created (and, potentially, for which we extended
  // the lifetime) to the subobject we're binding the reference to.
  if (!adjustments.empty()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitMaterializeTemporaryExpr: Adjustments");
    return {};
  }

  return makeAddrLValue(object, m->getType(), AlignmentSource::Decl);
}

LValue CIRGenFunction::emitCompoundLiteralLValue(const CompoundLiteralExpr *e) {
  if (e->isFileScope()) {
    cgm.errorNYI(e->getSourceRange(), "emitCompoundLiteralLValue: FileScope");
    return {};
  }

  if (e->getType()->isVariablyModifiedType()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCompoundLiteralLValue: VariablyModifiedType");
    return {};
  }

  Address declPtr = createMemTemp(e->getType(), getLoc(e->getSourceRange()),
                                  ".compoundliteral");
  const Expr *initExpr = e->getInitializer();
  LValue result = makeAddrLValue(declPtr, e->getType(), AlignmentSource::Decl);

  emitAnyExprToMem(initExpr, declPtr, e->getType().getQualifiers(),
                   /*Init*/ true);

  // Block-scope compound literals are destroyed at the end of the enclosing
  // scope in C.
  if (!getLangOpts().CPlusPlus && e->getType().isDestructedType()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCompoundLiteralLValue: non C++ DestructedType");
    return {};
  }

  return result;
}

LValue CIRGenFunction::emitCallExprLValue(const CallExpr *e) {
  RValue rv = emitCallExpr(e);

  if (!rv.isScalar()) {
    cgm.errorNYI(e->getSourceRange(), "emitCallExprLValue: non-scalar return");
    return {};
  }

  assert(e->getCallReturnType(getContext())->isReferenceType() &&
         "Can't have a scalar return unless the return type is a "
         "reference type!");

  return makeNaturalAlignPointeeAddrLValue(rv.getValue(), e->getType());
}

LValue CIRGenFunction::emitBinaryOperatorLValue(const BinaryOperator *e) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (e->getOpcode() == BO_Comma) {
    emitIgnoredExpr(e->getLHS());
    return emitLValue(e->getRHS());
  }

  if (e->getOpcode() == BO_PtrMemD || e->getOpcode() == BO_PtrMemI) {
    cgm.errorNYI(e->getSourceRange(), "member pointers");
    return {};
  }

  assert(e->getOpcode() == BO_Assign && "unexpected binary l-value");

  // Note that in all of these cases, __block variables need the RHS
  // evaluated first just in case the variable gets moved by the RHS.

  switch (CIRGenFunction::getEvaluationKind(e->getType())) {
  case cir::TEK_Scalar: {
    assert(!cir::MissingFeatures::objCLifetime());
    if (e->getLHS()->getType().getObjCLifetime() !=
        clang::Qualifiers::ObjCLifetime::OCL_None) {
      cgm.errorNYI(e->getSourceRange(), "objc lifetimes");
      return {};
    }

    RValue rv = emitAnyExpr(e->getRHS());
    LValue lv = emitLValue(e->getLHS());

    SourceLocRAIIObject loc{*this, getLoc(e->getSourceRange())};
    if (lv.isBitField())
      emitStoreThroughBitfieldLValue(rv, lv);
    else
      emitStoreThroughLValue(rv, lv);

    if (getLangOpts().OpenMP) {
      cgm.errorNYI(e->getSourceRange(), "openmp");
      return {};
    }

    return lv;
  }

  case cir::TEK_Complex: {
    return emitComplexAssignmentLValue(e);
  }

  case cir::TEK_Aggregate:
    cgm.errorNYI(e->getSourceRange(), "aggregate lvalues");
    return {};
  }
  llvm_unreachable("bad evaluation kind");
}

/// Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
RValue CIRGenFunction::emitAnyExpr(const Expr *e, AggValueSlot aggSlot) {
  switch (CIRGenFunction::getEvaluationKind(e->getType())) {
  case cir::TEK_Scalar:
    return RValue::get(emitScalarExpr(e));
  case cir::TEK_Complex:
    return RValue::getComplex(emitComplexExpr(e));
  case cir::TEK_Aggregate: {
    if (aggSlot.isIgnored())
      aggSlot = createAggTemp(e->getType(), getLoc(e->getSourceRange()),
                              getCounterAggTmpAsString());
    emitAggExpr(e, aggSlot);
    return aggSlot.asRValue();
  }
  }
  llvm_unreachable("bad evaluation kind");
}

// Detect the unusual situation where an inline version is shadowed by a
// non-inline version. In that case we should pick the external one
// everywhere. That's GCC behavior too.
static bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *fd) {
  for (const FunctionDecl *pd = fd; pd; pd = pd->getPreviousDecl())
    if (!pd->isInlineBuiltinDeclaration())
      return false;
  return true;
}

CIRGenCallee CIRGenFunction::emitDirectCallee(const GlobalDecl &gd) {
  const auto *fd = cast<FunctionDecl>(gd.getDecl());

  if (unsigned builtinID = fd->getBuiltinID()) {
    if (fd->getAttr<AsmLabelAttr>()) {
      cgm.errorNYI("AsmLabelAttr");
    }

    StringRef ident = fd->getName();
    std::string fdInlineName = (ident + ".inline").str();

    bool isPredefinedLibFunction =
        cgm.getASTContext().BuiltinInfo.isPredefinedLibFunction(builtinID);
    // Assume nobuiltins everywhere until we actually read the attributes.
    bool hasAttributeNoBuiltin = true;
    assert(!cir::MissingFeatures::attributeNoBuiltin());

    // When directing calling an inline builtin, call it through it's mangled
    // name to make it clear it's not the actual builtin.
    auto fn = cast<cir::FuncOp>(curFn);
    if (fn.getName() != fdInlineName && onlyHasInlineBuiltinDeclaration(fd)) {
      cgm.errorNYI("Inline only builtin function calls");
    }

    // Replaceable builtins provide their own implementation of a builtin. If we
    // are in an inline builtin implementation, avoid trivial infinite
    // recursion. Honor __attribute__((no_builtin("foo"))) or
    // __attribute__((no_builtin)) on the current function unless foo is
    // not a predefined library function which means we must generate the
    // builtin no matter what.
    else if (!isPredefinedLibFunction || !hasAttributeNoBuiltin)
      return CIRGenCallee::forBuiltin(builtinID, fd);
  }

  cir::FuncOp callee = emitFunctionDeclPointer(cgm, gd);

  assert(!cir::MissingFeatures::hip());

  return CIRGenCallee::forDirect(callee, gd);
}

RValue CIRGenFunction::getUndefRValue(QualType ty) {
  if (ty->isVoidType())
    return RValue::get(nullptr);

  cgm.errorNYI("unsupported type for undef rvalue");
  return RValue::get(nullptr);
}

RValue CIRGenFunction::emitCall(clang::QualType calleeTy,
                                const CIRGenCallee &origCallee,
                                const clang::CallExpr *e,
                                ReturnValueSlot returnValue) {
  // Get the actual function type. The callee type will always be a pointer to
  // function type or a block pointer type.
  assert(calleeTy->isFunctionPointerType() &&
         "Callee must have function pointer type!");

  calleeTy = getContext().getCanonicalType(calleeTy);
  auto pointeeTy = cast<PointerType>(calleeTy)->getPointeeType();

  CIRGenCallee callee = origCallee;

  if (getLangOpts().CPlusPlus)
    assert(!cir::MissingFeatures::sanitizers());

  const auto *fnType = cast<FunctionType>(pointeeTy);

  assert(!cir::MissingFeatures::sanitizers());

  CallArgList args;
  assert(!cir::MissingFeatures::opCallArgEvaluationOrder());

  emitCallArgs(args, dyn_cast<FunctionProtoType>(fnType), e->arguments(),
               e->getDirectCallee());

  const CIRGenFunctionInfo &funcInfo =
      cgm.getTypes().arrangeFreeFunctionCall(args, fnType);

  // C99 6.5.2.2p6:
  //   If the expression that denotes the called function has a type that does
  //   not include a prototype, [the default argument promotions are performed].
  //   If the number of arguments does not equal the number of parameters, the
  //   behavior is undefined. If the function is defined with a type that
  //   includes a prototype, and either the prototype ends with an ellipsis (,
  //   ...) or the types of the arguments after promotion are not compatible
  //   with the types of the parameters, the behavior is undefined. If the
  //   function is defined with a type that does not include a prototype, and
  //   the types of the arguments after promotion are not compatible with those
  //   of the parameters after promotion, the behavior is undefined [except in
  //   some trivial cases].
  // That is, in the general case, we should assume that a call through an
  // unprototyped function type works like a *non-variadic* call. The way we
  // make this work is to cast to the exxact type fo the promoted arguments.
  if (isa<FunctionNoProtoType>(fnType)) {
    assert(!cir::MissingFeatures::opCallChain());
    assert(!cir::MissingFeatures::addressSpace());
    cir::FuncType calleeTy = getTypes().getFunctionType(funcInfo);
    // get non-variadic function type
    calleeTy = cir::FuncType::get(calleeTy.getInputs(),
                                  calleeTy.getReturnType(), false);
    auto calleePtrTy = cir::PointerType::get(calleeTy);

    mlir::Operation *fn = callee.getFunctionPointer();
    mlir::Value addr;
    if (auto funcOp = mlir::dyn_cast<cir::FuncOp>(fn)) {
      addr = builder.create<cir::GetGlobalOp>(
          getLoc(e->getSourceRange()),
          cir::PointerType::get(funcOp.getFunctionType()), funcOp.getSymName());
    } else {
      addr = fn->getResult(0);
    }

    fn = builder.createBitcast(addr, calleePtrTy).getDefiningOp();
    callee.setFunctionPointer(fn);
  }

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  assert(!cir::MissingFeatures::hip());
  assert(!cir::MissingFeatures::opCallMustTail());

  cir::CIRCallOpInterface callOp;
  RValue callResult = emitCall(funcInfo, callee, returnValue, args, &callOp,
                               getLoc(e->getExprLoc()));

  assert(!cir::MissingFeatures::generateDebugInfo());

  return callResult;
}

CIRGenCallee CIRGenFunction::emitCallee(const clang::Expr *e) {
  e = e->IgnoreParens();

  // Look through function-to-pointer decay.
  if (const auto *implicitCast = dyn_cast<ImplicitCastExpr>(e)) {
    if (implicitCast->getCastKind() == CK_FunctionToPointerDecay ||
        implicitCast->getCastKind() == CK_BuiltinFnToFnPtr) {
      return emitCallee(implicitCast->getSubExpr());
    }
    // When performing an indirect call through a function pointer lvalue, the
    // function pointer lvalue is implicitly converted to an rvalue through an
    // lvalue-to-rvalue conversion.
    assert(implicitCast->getCastKind() == CK_LValueToRValue &&
           "unexpected implicit cast on function pointers");
  } else if (const auto *declRef = dyn_cast<DeclRefExpr>(e)) {
    // Resolve direct calls.
    const auto *funcDecl = cast<FunctionDecl>(declRef->getDecl());
    return emitDirectCallee(funcDecl);
  } else if (isa<MemberExpr>(e)) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCallee: call to member function is NYI");
    return {};
  } else if (auto *pde = dyn_cast<CXXPseudoDestructorExpr>(e)) {
    return CIRGenCallee::forPseudoDestructor(pde);
  }

  // Otherwise, we have an indirect reference.
  mlir::Value calleePtr;
  QualType functionType;
  if (const auto *ptrType = e->getType()->getAs<clang::PointerType>()) {
    calleePtr = emitScalarExpr(e);
    functionType = ptrType->getPointeeType();
  } else {
    functionType = e->getType();
    calleePtr = emitLValue(e).getPointer();
  }
  assert(functionType->isFunctionType());

  GlobalDecl gd;
  if (const auto *vd =
          dyn_cast_or_null<VarDecl>(e->getReferencedDeclOfCallee()))
    gd = GlobalDecl(vd);

  CIRGenCalleeInfo calleeInfo(functionType->getAs<FunctionProtoType>(), gd);
  CIRGenCallee callee(calleeInfo, calleePtr.getDefiningOp());
  return callee;
}

RValue CIRGenFunction::emitCallExpr(const clang::CallExpr *e,
                                    ReturnValueSlot returnValue) {
  assert(!cir::MissingFeatures::objCBlocks());

  if (const auto *ce = dyn_cast<CXXMemberCallExpr>(e))
    return emitCXXMemberCallExpr(ce, returnValue);

  if (isa<CUDAKernelCallExpr>(e)) {
    cgm.errorNYI(e->getSourceRange(), "call to CUDA kernel");
    return RValue::get(nullptr);
  }

  if (const auto *operatorCall = dyn_cast<CXXOperatorCallExpr>(e)) {
    // If the callee decl is a CXXMethodDecl, we need to emit this as a C++
    // operator member call.
    if (const CXXMethodDecl *md =
            dyn_cast_or_null<CXXMethodDecl>(operatorCall->getCalleeDecl()))
      return emitCXXOperatorMemberCallExpr(operatorCall, md, returnValue);
    // A CXXOperatorCallExpr is created even for explicit object methods, but
    // these should be treated like static function calls. Fall through to do
    // that.
  }

  CIRGenCallee callee = emitCallee(e->getCallee());

  if (callee.isBuiltin())
    return emitBuiltinExpr(callee.getBuiltinDecl(), callee.getBuiltinID(), e,
                           returnValue);

  if (callee.isPseudoDestructor())
    return emitCXXPseudoDestructorExpr(callee.getPseudoDestructorExpr());

  return emitCall(e->getCallee()->getType(), callee, e, returnValue);
}

/// Emit code to compute the specified expression, ignoring the result.
void CIRGenFunction::emitIgnoredExpr(const Expr *e) {
  if (e->isPRValue()) {
    assert(!cir::MissingFeatures::aggValueSlot());
    emitAnyExpr(e);
    return;
  }

  // Just emit it as an l-value and drop the result.
  emitLValue(e);
}

Address CIRGenFunction::emitArrayToPointerDecay(const Expr *e,
                                                LValueBaseInfo *baseInfo) {
  assert(!cir::MissingFeatures::opTBAA());
  assert(e->getType()->isArrayType() &&
         "Array to pointer decay must have array source type!");

  // Expressions of array type can't be bitfields or vector elements.
  LValue lv = emitLValue(e);
  Address addr = lv.getAddress();

  // If the array type was an incomplete type, we need to make sure
  // the decay ends up being the right type.
  auto lvalueAddrTy = mlir::cast<cir::PointerType>(addr.getPointer().getType());

  if (e->getType()->isVariableArrayType())
    return addr;

  [[maybe_unused]] auto pointeeTy =
      mlir::cast<cir::ArrayType>(lvalueAddrTy.getPointee());

  [[maybe_unused]] mlir::Type arrayTy = convertType(e->getType());
  assert(mlir::isa<cir::ArrayType>(arrayTy) && "expected array");
  assert(pointeeTy == arrayTy);

  // The result of this decay conversion points to an array element within the
  // base lvalue. However, since TBAA currently does not support representing
  // accesses to elements of member arrays, we conservatively represent accesses
  // to the pointee object as if it had no any base lvalue specified.
  // TODO: Support TBAA for member arrays.
  QualType eltType = e->getType()->castAsArrayTypeUnsafe()->getElementType();
  assert(!cir::MissingFeatures::opTBAA());

  mlir::Value ptr = builder.maybeBuildArrayDecay(
      cgm.getLoc(e->getSourceRange()), addr.getPointer(),
      convertTypeForMem(eltType));
  return Address(ptr, addr.getAlignment());
}

/// Given the address of a temporary variable, produce an r-value of its type.
RValue CIRGenFunction::convertTempToRValue(Address addr, clang::QualType type,
                                           clang::SourceLocation loc) {
  LValue lvalue = makeAddrLValue(addr, type, AlignmentSource::Decl);
  switch (getEvaluationKind(type)) {
  case cir::TEK_Complex:
    cgm.errorNYI(loc, "convertTempToRValue: complex type");
    return RValue::get(nullptr);
  case cir::TEK_Aggregate:
    cgm.errorNYI(loc, "convertTempToRValue: aggregate type");
    return RValue::get(nullptr);
  case cir::TEK_Scalar:
    return RValue::get(emitLoadOfScalar(lvalue, loc));
  }
  llvm_unreachable("bad evaluation kind");
}

/// Emit an `if` on a boolean condition, filling `then` and `else` into
/// appropriated regions.
mlir::LogicalResult CIRGenFunction::emitIfOnBoolExpr(const Expr *cond,
                                                     const Stmt *thenS,
                                                     const Stmt *elseS) {
  mlir::Location thenLoc = getLoc(thenS->getSourceRange());
  std::optional<mlir::Location> elseLoc;
  if (elseS)
    elseLoc = getLoc(elseS->getSourceRange());

  mlir::LogicalResult resThen = mlir::success(), resElse = mlir::success();
  emitIfOnBoolExpr(
      cond, /*thenBuilder=*/
      [&](mlir::OpBuilder &, mlir::Location) {
        LexicalScope lexScope{*this, thenLoc, builder.getInsertionBlock()};
        resThen = emitStmt(thenS, /*useCurrentScope=*/true);
      },
      thenLoc,
      /*elseBuilder=*/
      [&](mlir::OpBuilder &, mlir::Location) {
        assert(elseLoc && "Invalid location for elseS.");
        LexicalScope lexScope{*this, *elseLoc, builder.getInsertionBlock()};
        resElse = emitStmt(elseS, /*useCurrentScope=*/true);
      },
      elseLoc);

  return mlir::LogicalResult::success(resThen.succeeded() &&
                                      resElse.succeeded());
}

/// Emit an `if` on a boolean condition, filling `then` and `else` into
/// appropriated regions.
cir::IfOp CIRGenFunction::emitIfOnBoolExpr(
    const clang::Expr *cond, BuilderCallbackRef thenBuilder,
    mlir::Location thenLoc, BuilderCallbackRef elseBuilder,
    std::optional<mlir::Location> elseLoc) {
  // Attempt to be as accurate as possible with IfOp location, generate
  // one fused location that has either 2 or 4 total locations, depending
  // on else's availability.
  SmallVector<mlir::Location, 2> ifLocs{thenLoc};
  if (elseLoc)
    ifLocs.push_back(*elseLoc);
  mlir::Location loc = mlir::FusedLoc::get(&getMLIRContext(), ifLocs);

  // Emit the code with the fully general case.
  mlir::Value condV = emitOpOnBoolExpr(loc, cond);
  return builder.create<cir::IfOp>(loc, condV, elseLoc.has_value(),
                                   /*thenBuilder=*/thenBuilder,
                                   /*elseBuilder=*/elseBuilder);
}

/// TODO(cir): see EmitBranchOnBoolExpr for extra ideas).
mlir::Value CIRGenFunction::emitOpOnBoolExpr(mlir::Location loc,
                                             const Expr *cond) {
  assert(!cir::MissingFeatures::pgoUse());
  assert(!cir::MissingFeatures::generateDebugInfo());
  cond = cond->IgnoreParens();

  // In LLVM the condition is reversed here for efficient codegen.
  // This should be done in CIR prior to LLVM lowering, if we do now
  // we can make CIR based diagnostics misleading.
  //  cir.ternary(!x, t, f) -> cir.ternary(x, f, t)
  assert(!cir::MissingFeatures::shouldReverseUnaryCondOnBoolExpr());

  if (const ConditionalOperator *condOp = dyn_cast<ConditionalOperator>(cond)) {
    Expr *trueExpr = condOp->getTrueExpr();
    Expr *falseExpr = condOp->getFalseExpr();
    mlir::Value condV = emitOpOnBoolExpr(loc, condOp->getCond());

    mlir::Value ternaryOpRes =
        builder
            .create<cir::TernaryOp>(
                loc, condV, /*thenBuilder=*/
                [this, trueExpr](mlir::OpBuilder &b, mlir::Location loc) {
                  mlir::Value lhs = emitScalarExpr(trueExpr);
                  b.create<cir::YieldOp>(loc, lhs);
                },
                /*elseBuilder=*/
                [this, falseExpr](mlir::OpBuilder &b, mlir::Location loc) {
                  mlir::Value rhs = emitScalarExpr(falseExpr);
                  b.create<cir::YieldOp>(loc, rhs);
                })
            .getResult();

    return emitScalarConversion(ternaryOpRes, condOp->getType(),
                                getContext().BoolTy, condOp->getExprLoc());
  }

  if (isa<CXXThrowExpr>(cond)) {
    cgm.errorNYI("NYI");
    return createDummyValue(loc, cond->getType());
  }

  // If the branch has a condition wrapped by __builtin_unpredictable,
  // create metadata that specifies that the branch is unpredictable.
  // Don't bother if not optimizing because that metadata would not be used.
  assert(!cir::MissingFeatures::insertBuiltinUnpredictable());

  // Emit the code with the fully general case.
  return evaluateExprAsBool(cond);
}

mlir::Value CIRGenFunction::emitAlloca(StringRef name, mlir::Type ty,
                                       mlir::Location loc, CharUnits alignment,
                                       bool insertIntoFnEntryBlock,
                                       mlir::Value arraySize) {
  mlir::Block *entryBlock = insertIntoFnEntryBlock
                                ? getCurFunctionEntryBlock()
                                : curLexScope->getEntryBlock();

  // If this is an alloca in the entry basic block of a cir.try and there's
  // a surrounding cir.scope, make sure the alloca ends up in the surrounding
  // scope instead. This is necessary in order to guarantee all SSA values are
  // reachable during cleanups.
  assert(!cir::MissingFeatures::tryOp());

  return emitAlloca(name, ty, loc, alignment,
                    builder.getBestAllocaInsertPoint(entryBlock), arraySize);
}

mlir::Value CIRGenFunction::emitAlloca(StringRef name, mlir::Type ty,
                                       mlir::Location loc, CharUnits alignment,
                                       mlir::OpBuilder::InsertPoint ip,
                                       mlir::Value arraySize) {
  // CIR uses its own alloca address space rather than follow the target data
  // layout like original CodeGen. The data layout awareness should be done in
  // the lowering pass instead.
  assert(!cir::MissingFeatures::addressSpace());
  cir::PointerType localVarPtrTy = builder.getPointerTo(ty);
  mlir::IntegerAttr alignIntAttr = cgm.getSize(alignment);

  mlir::Value addr;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(ip);
    addr = builder.createAlloca(loc, /*addr type*/ localVarPtrTy,
                                /*var type*/ ty, name, alignIntAttr);
    assert(!cir::MissingFeatures::astVarDeclInterface());
  }
  return addr;
}

// Note: this function also emit constructor calls to support a MSVC extensions
// allowing explicit constructor function call.
RValue CIRGenFunction::emitCXXMemberCallExpr(const CXXMemberCallExpr *ce,
                                             ReturnValueSlot returnValue) {
  const Expr *callee = ce->getCallee()->IgnoreParens();

  if (isa<BinaryOperator>(callee)) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberCallExpr: C++ binary operator");
    return RValue::get(nullptr);
  }

  const auto *me = cast<MemberExpr>(callee);
  const auto *md = cast<CXXMethodDecl>(me->getMemberDecl());

  if (md->isStatic()) {
    cgm.errorNYI(ce->getSourceRange(), "emitCXXMemberCallExpr: static method");
    return RValue::get(nullptr);
  }

  bool hasQualifier = me->hasQualifier();
  NestedNameSpecifier qualifier = me->getQualifier();
  bool isArrow = me->isArrow();
  const Expr *base = me->getBase();

  return emitCXXMemberOrOperatorMemberCallExpr(
      ce, md, returnValue, hasQualifier, qualifier, isArrow, base);
}

void CIRGenFunction::emitCXXConstructExpr(const CXXConstructExpr *e,
                                          AggValueSlot dest) {
  assert(!dest.isIgnored() && "Must have a destination!");
  const CXXConstructorDecl *cd = e->getConstructor();

  // If we require zero initialization before (or instead of) calling the
  // constructor, as can be the case with a non-user-provided default
  // constructor, emit the zero initialization now, unless destination is
  // already zeroed.
  if (e->requiresZeroInitialization() && !dest.isZeroed()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCXXConstructExpr: requires initialization");
    return;
  }

  // If this is a call to a trivial default constructor:
  // In LLVM: do nothing.
  // In CIR: emit as a regular call, other later passes should lower the
  // ctor call into trivial initialization.

  // Elide the constructor if we're constructing from a temporary
  if (getLangOpts().ElideConstructors && e->isElidable()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCXXConstructExpr: elidable constructor");
    return;
  }

  if (const ArrayType *arrayType = getContext().getAsArrayType(e->getType())) {
    assert(!cir::MissingFeatures::sanitizers());
    emitCXXAggrConstructorCall(cd, arrayType, dest.getAddress(), e, false);
  } else {

    clang::CXXCtorType type = Ctor_Complete;
    bool forVirtualBase = false;
    bool delegating = false;

    switch (e->getConstructionKind()) {
    case CXXConstructionKind::Complete:
      type = Ctor_Complete;
      break;
    case CXXConstructionKind::Delegating:
      // We should be emitting a constructor; GlobalDecl will assert this
      type = curGD.getCtorType();
      delegating = true;
      break;
    case CXXConstructionKind::VirtualBase:
      forVirtualBase = true;
      [[fallthrough]];
    case CXXConstructionKind::NonVirtualBase:
      type = Ctor_Base;
      break;
    }

    emitCXXConstructorCall(cd, type, forVirtualBase, delegating, dest, e);
  }
}

RValue CIRGenFunction::emitReferenceBindingToExpr(const Expr *e) {
  // Emit the expression as an lvalue.
  LValue lv = emitLValue(e);
  assert(lv.isSimple());
  mlir::Value value = lv.getPointer();

  assert(!cir::MissingFeatures::sanitizers());

  return RValue::get(value);
}

Address CIRGenFunction::emitLoadOfReference(LValue refLVal, mlir::Location loc,
                                            LValueBaseInfo *pointeeBaseInfo) {
  if (refLVal.isVolatile())
    cgm.errorNYI(loc, "load of volatile reference");

  cir::LoadOp load =
      builder.create<cir::LoadOp>(loc, refLVal.getAddress().getElementType(),
                                  refLVal.getAddress().getPointer());

  assert(!cir::MissingFeatures::opTBAA());

  QualType pointeeType = refLVal.getType()->getPointeeType();
  CharUnits align = cgm.getNaturalTypeAlignment(pointeeType, pointeeBaseInfo);
  return Address(load, convertTypeForMem(pointeeType), align);
}

LValue CIRGenFunction::emitLoadOfReferenceLValue(Address refAddr,
                                                 mlir::Location loc,
                                                 QualType refTy,
                                                 AlignmentSource source) {
  LValue refLVal = makeAddrLValue(refAddr, refTy, LValueBaseInfo(source));
  LValueBaseInfo pointeeBaseInfo;
  assert(!cir::MissingFeatures::opTBAA());
  Address pointeeAddr = emitLoadOfReference(refLVal, loc, &pointeeBaseInfo);
  return makeAddrLValue(pointeeAddr, refLVal.getType()->getPointeeType(),
                        pointeeBaseInfo);
}

void CIRGenFunction::emitTrap(mlir::Location loc, bool createNewBlock) {
  cir::TrapOp::create(builder, loc);
  if (createNewBlock)
    builder.createBlock(builder.getBlock()->getParent());
}

void CIRGenFunction::emitUnreachable(clang::SourceLocation loc,
                                     bool createNewBlock) {
  assert(!cir::MissingFeatures::sanitizers());
  cir::UnreachableOp::create(builder, getLoc(loc));
  if (createNewBlock)
    builder.createBlock(builder.getBlock()->getParent());
}

mlir::Value CIRGenFunction::createDummyValue(mlir::Location loc,
                                             clang::QualType qt) {
  mlir::Type t = convertType(qt);
  CharUnits alignment = getContext().getTypeAlignInChars(qt);
  return builder.createDummyValue(loc, t, alignment);
}

//===----------------------------------------------------------------------===//
// CIR builder helpers
//===----------------------------------------------------------------------===//

Address CIRGenFunction::createMemTemp(QualType ty, mlir::Location loc,
                                      const Twine &name, Address *alloca,
                                      mlir::OpBuilder::InsertPoint ip) {
  // FIXME: Should we prefer the preferred type alignment here?
  return createMemTemp(ty, getContext().getTypeAlignInChars(ty), loc, name,
                       alloca, ip);
}

Address CIRGenFunction::createMemTemp(QualType ty, CharUnits align,
                                      mlir::Location loc, const Twine &name,
                                      Address *alloca,
                                      mlir::OpBuilder::InsertPoint ip) {
  Address result = createTempAlloca(convertTypeForMem(ty), align, loc, name,
                                    /*ArraySize=*/nullptr, alloca, ip);
  if (ty->isConstantMatrixType()) {
    assert(!cir::MissingFeatures::matrixType());
    cgm.errorNYI(loc, "temporary matrix value");
  }
  return result;
}

/// This creates a alloca and inserts it into the entry block of the
/// current region.
Address CIRGenFunction::createTempAllocaWithoutCast(
    mlir::Type ty, CharUnits align, mlir::Location loc, const Twine &name,
    mlir::Value arraySize, mlir::OpBuilder::InsertPoint ip) {
  cir::AllocaOp alloca = ip.isSet()
                             ? createTempAlloca(ty, loc, name, ip, arraySize)
                             : createTempAlloca(ty, loc, name, arraySize);
  alloca.setAlignmentAttr(cgm.getSize(align));
  return Address(alloca, ty, align);
}

/// This creates a alloca and inserts it into the entry block. The alloca is
/// casted to default address space if necessary.
Address CIRGenFunction::createTempAlloca(mlir::Type ty, CharUnits align,
                                         mlir::Location loc, const Twine &name,
                                         mlir::Value arraySize,
                                         Address *allocaAddr,
                                         mlir::OpBuilder::InsertPoint ip) {
  Address alloca =
      createTempAllocaWithoutCast(ty, align, loc, name, arraySize, ip);
  if (allocaAddr)
    *allocaAddr = alloca;
  mlir::Value v = alloca.getPointer();
  // Alloca always returns a pointer in alloca address space, which may
  // be different from the type defined by the language. For example,
  // in C++ the auto variables are in the default address space. Therefore
  // cast alloca to the default address space when necessary.
  assert(!cir::MissingFeatures::addressSpace());
  return Address(v, ty, align);
}

/// This creates an alloca and inserts it into the entry block if \p ArraySize
/// is nullptr, otherwise inserts it at the current insertion point of the
/// builder.
cir::AllocaOp CIRGenFunction::createTempAlloca(mlir::Type ty,
                                               mlir::Location loc,
                                               const Twine &name,
                                               mlir::Value arraySize,
                                               bool insertIntoFnEntryBlock) {
  return mlir::cast<cir::AllocaOp>(emitAlloca(name.str(), ty, loc, CharUnits(),
                                              insertIntoFnEntryBlock, arraySize)
                                       .getDefiningOp());
}

/// This creates an alloca and inserts it into the provided insertion point
cir::AllocaOp CIRGenFunction::createTempAlloca(mlir::Type ty,
                                               mlir::Location loc,
                                               const Twine &name,
                                               mlir::OpBuilder::InsertPoint ip,
                                               mlir::Value arraySize) {
  assert(ip.isSet() && "Insertion point is not set");
  return mlir::cast<cir::AllocaOp>(
      emitAlloca(name.str(), ty, loc, CharUnits(), ip, arraySize)
          .getDefiningOp());
}

/// Try to emit a reference to the given value without producing it as
/// an l-value.  For many cases, this is just an optimization, but it avoids
/// us needing to emit global copies of variables if they're named without
/// triggering a formal use in a context where we can't emit a direct
/// reference to them, for instance if a block or lambda or a member of a
/// local class uses a const int variable or constexpr variable from an
/// enclosing function.
///
/// For named members of enums, this is the only way they are emitted.
CIRGenFunction::ConstantEmission
CIRGenFunction::tryEmitAsConstant(const DeclRefExpr *refExpr) {
  const ValueDecl *value = refExpr->getDecl();

  // There is a lot more to do here, but for now only EnumConstantDecl is
  // supported.
  assert(!cir::MissingFeatures::tryEmitAsConstant());

  // The value needs to be an enum constant or a constant variable.
  if (!isa<EnumConstantDecl>(value))
    return ConstantEmission();

  Expr::EvalResult result;
  if (!refExpr->EvaluateAsRValue(result, getContext()))
    return ConstantEmission();

  QualType resultType = refExpr->getType();

  // As long as we're only handling EnumConstantDecl, there should be no
  // side-effects.
  assert(!result.HasSideEffects);

  // Emit as a constant.
  // FIXME(cir): have emitAbstract build a TypedAttr instead (this requires
  // somewhat heavy refactoring...)
  mlir::Attribute c = ConstantEmitter(*this).emitAbstract(
      refExpr->getLocation(), result.Val, resultType);
  mlir::TypedAttr cstToEmit = mlir::dyn_cast_if_present<mlir::TypedAttr>(c);
  assert(cstToEmit && "expected a typed attribute");

  assert(!cir::MissingFeatures::generateDebugInfo());

  return ConstantEmission::forValue(cstToEmit);
}

CIRGenFunction::ConstantEmission
CIRGenFunction::tryEmitAsConstant(const MemberExpr *me) {
  if (DeclRefExpr *dre = tryToConvertMemberExprToDeclRefExpr(*this, me))
    return tryEmitAsConstant(dre);
  return ConstantEmission();
}

mlir::Value CIRGenFunction::emitScalarConstant(
    const CIRGenFunction::ConstantEmission &constant, Expr *e) {
  assert(constant && "not a constant");
  if (constant.isReference()) {
    cgm.errorNYI(e->getSourceRange(), "emitScalarConstant: reference");
    return {};
  }
  return builder.getConstant(getLoc(e->getSourceRange()), constant.getValue());
}

/// An LValue is a candidate for having its loads and stores be made atomic if
/// we are operating under /volatile:ms *and* the LValue itself is volatile and
/// performing such an operation can be performed without a libcall.
bool CIRGenFunction::isLValueSuitableForInlineAtomic(LValue lv) {
  if (!cgm.getLangOpts().MSVolatile)
    return false;

  cgm.errorNYI("LValueSuitableForInlineAtomic LangOpts MSVolatile");
  return false;
}
