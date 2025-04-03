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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

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
    cgm.errorNYI(e->getSourceRange(), "UnaryOp dereference");
    return LValue();
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
