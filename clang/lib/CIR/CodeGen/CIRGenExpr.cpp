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
#include "CIRGenFunction.h"
#include "CIRGenValue.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

/// Given an expression of pointer type, try to
/// derive a more accurate bound on the alignment of the pointer.
Address CIRGenFunction::emitPointerWithAlignment(const Expr *expr) {
  // We allow this with ObjC object pointers because of fragile ABIs.
  assert(expr->getType()->isPointerType() ||
         expr->getType()->isObjCObjectPointerType());
  expr = expr->IgnoreParens();

  // Casts:
  if (auto const *ce = dyn_cast<CastExpr>(expr)) {
    if (auto const *ece = dyn_cast<ExplicitCastExpr>(ce)) {
      cgm.errorNYI(expr->getSourceRange(),
                   "emitPointerWithAlignment: explicit cast");
      return Address::invalid();
    }

    switch (ce->getCastKind()) {
    // Non-converting casts (but not C's implicit conversion from void*).
    case CK_BitCast:
    case CK_NoOp:
    case CK_AddressSpaceConversion: {
      cgm.errorNYI(expr->getSourceRange(),
                   "emitPointerWithAlignment: noop cast");
      return Address::invalid();
    } break;

    // Array-to-pointer decay. TODO(cir): BaseInfo and TBAAInfo.
    case CK_ArrayToPointerDecay: {
      cgm.errorNYI(expr->getSourceRange(),
                   "emitPointerWithAlignment: array-to-pointer decay");
      return Address::invalid();
    }

    case CK_UncheckedDerivedToBase:
    case CK_DerivedToBase: {
      cgm.errorNYI(expr->getSourceRange(),
                   "emitPointerWithAlignment: derived-to-base cast");
      return Address::invalid();
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
      cgm.errorNYI(expr->getSourceRange(), "emitPointerWithAlignment: unary &");
      return Address::invalid();
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
      emitScalarExpr(expr), expr->getType()->getPointeeType(), CharUnits());
}

void CIRGenFunction::emitStoreThroughLValue(RValue src, LValue dst,
                                            bool isInit) {
  if (!dst.isSimple()) {
    cgm.errorNYI(dst.getPointer().getLoc(),
                 "emitStoreThroughLValue: non-simple lvalue");
    return;
  }

  assert(!cir::MissingFeatures::opLoadStoreObjC());

  assert(src.isScalar() && "Can't emit an aggregate store with this method");
  emitStoreOfScalar(src.getScalarVal(), dst, isInit);
}

void CIRGenFunction::emitStoreOfScalar(mlir::Value value, Address addr,
                                       bool isVolatile, QualType ty,
                                       bool isInit, bool isNontemporal) {
  assert(!cir::MissingFeatures::opLoadStoreThreadLocal());

  if (ty->getAs<clang::VectorType>()) {
    cgm.errorNYI(addr.getPointer().getLoc(), "emitStoreOfScalar vector type");
    return;
  }

  value = emitToMemory(value, ty);

  assert(!cir::MissingFeatures::opLoadStoreAtomic());

  // Update the alloca with more info on initialization.
  assert(addr.getPointer() && "expected pointer to exist");
  auto srcAlloca =
      dyn_cast_or_null<cir::AllocaOp>(addr.getPointer().getDefiningOp());
  if (currVarDecl && srcAlloca) {
    const VarDecl *vd = currVarDecl;
    assert(vd && "VarDecl expected");
    if (vd->hasInit())
      srcAlloca.setInitAttr(mlir::UnitAttr::get(&getMLIRContext()));
  }

  assert(currSrcLoc && "must pass in source location");
  builder.createStore(*currSrcLoc, value, addr.getPointer() /*, isVolatile*/);

  if (isNontemporal) {
    cgm.errorNYI(addr.getPointer().getLoc(), "emitStoreOfScalar nontemporal");
    return;
  }

  assert(!cir::MissingFeatures::opTBAA());
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

  mlir::Value ptr = addr.getPointer();
  if (mlir::isa<cir::VoidType>(eltTy))
    cgm.errorNYI(loc, "emitLoadOfScalar: void type");

  mlir::Value loadOp = builder.CIRBaseBuilderTy::createLoad(
      getLoc(loc), ptr, false /*isVolatile*/);

  return loadOp;
}

/// Given an expression that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result as an rvalue,
/// returning the rvalue.
RValue CIRGenFunction::emitLoadOfLValue(LValue lv, SourceLocation loc) {
  assert(!lv.getType()->isFunctionType());
  assert(!(lv.getType()->isConstantMatrixType()) && "not implemented");

  if (lv.isSimple())
    return RValue::get(emitLoadOfScalar(lv, loc));

  cgm.errorNYI(loc, "emitLoadOfLValue");
  return RValue::get(nullptr);
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
      cgm.errorNYI(vd->getSourceRange(), "emitDeclRefLValue: global variable");

    Address addr = Address::invalid();

    // The variable should generally be present in the local decl map.
    auto iter = localDeclMap.find(vd);
    if (iter != localDeclMap.end()) {
      addr = iter->second;
    } else {
      // Otherwise, it might be static local we haven't emitted yet for some
      // reason; most likely, because it's in an outer function.
      cgm.errorNYI(vd->getSourceRange(), "emitDeclRefLValue: static local");
    }

    return LValue::makeAddr(addr, ty);
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

    assert(!cir::MissingFeatures::lvalueBaseInfo());
    assert(!cir::MissingFeatures::opTBAA());
    Address addr = emitPointerWithAlignment(e->getSubExpr());

    // Tag 'load' with deref attribute.
    // FIXME: This misses some derefence cases and has problematic interactions
    // with other operators.
    if (auto loadOp =
            dyn_cast<cir::LoadOp>(addr.getPointer().getDefiningOp())) {
      loadOp.setIsDerefAttr(mlir::UnitAttr::get(&getMLIRContext()));
    }

    LValue lv = LValue::makeAddr(addr, t);
    assert(!cir::MissingFeatures::addressSpace());
    assert(!cir::MissingFeatures::setNonGC());
    return lv;
  }
  case UO_Real:
  case UO_Imag: {
    cgm.errorNYI(e->getSourceRange(), "UnaryOp real/imag");
    return LValue();
  }
  case UO_PreInc:
  case UO_PreDec: {
    bool isInc = e->isIncrementOp();
    LValue lv = emitLValue(e->getSubExpr());

    assert(e->isPrefix() && "Prefix operator in unexpected state!");

    if (e->getType()->isAnyComplexType()) {
      cgm.errorNYI(e->getSourceRange(), "UnaryOp complex inc/dec");
      lv = LValue();
    } else {
      emitScalarPrePostIncDec(e, lv, isInc, /*isPre=*/true);
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
    if (lv.isBitField()) {
      cgm.errorNYI(e->getSourceRange(), "bitfields");
      return {};
    }
    emitStoreThroughLValue(rv, lv);

    if (getLangOpts().OpenMP) {
      cgm.errorNYI(e->getSourceRange(), "openmp");
      return {};
    }

    return lv;
  }

  case cir::TEK_Complex: {
    assert(!cir::MissingFeatures::complexType());
    cgm.errorNYI(e->getSourceRange(), "complex l-values");
    return {};
  }
  case cir::TEK_Aggregate:
    cgm.errorNYI(e->getSourceRange(), "aggregate lvalues");
    return {};
  }
  llvm_unreachable("bad evaluation kind");
}

/// Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
RValue CIRGenFunction::emitAnyExpr(const Expr *e) {
  switch (CIRGenFunction::getEvaluationKind(e->getType())) {
  case cir::TEK_Scalar:
    return RValue::get(emitScalarExpr(e));
  case cir::TEK_Complex:
    cgm.errorNYI(e->getSourceRange(), "emitAnyExpr: complex type");
    return RValue::get(nullptr);
  case cir::TEK_Aggregate:
    cgm.errorNYI(e->getSourceRange(), "emitAnyExpr: aggregate type");
    return RValue::get(nullptr);
  }
  llvm_unreachable("bad evaluation kind");
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

  if (isa<ConditionalOperator>(cond)) {
    cgm.errorNYI(cond->getExprLoc(), "Ternary NYI");
    assert(!cir::MissingFeatures::ternaryOp());
    return createDummyValue(loc, cond->getType());
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

mlir::Value CIRGenFunction::createDummyValue(mlir::Location loc,
                                             clang::QualType qt) {
  mlir::Type t = convertType(qt);
  CharUnits alignment = getContext().getTypeAlignInChars(qt);
  return builder.createDummyValue(loc, t, alignment);
}

/// This creates an alloca and inserts it into the entry block if
/// \p insertIntoFnEntryBlock is true, otherwise it inserts it at the current
/// insertion point of the builder.
Address CIRGenFunction::createTempAlloca(mlir::Type ty, CharUnits align,
                                         mlir::Location loc, const Twine &name,
                                         bool insertIntoFnEntryBlock) {
  mlir::Value alloca =
      emitAlloca(name.str(), ty, loc, align, insertIntoFnEntryBlock);
  return Address(alloca, ty, align);
}
