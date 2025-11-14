//===----- CGCoroutine.cpp - Emit CIR Code for C++ coroutines -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of coroutines.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

struct clang::CIRGen::CGCoroData {
  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  cir::CallOp coroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value coroBegin = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
CIRGenFunction::CGCoroInfo::CGCoroInfo() {}
CIRGenFunction::CGCoroInfo::~CGCoroInfo() {}

namespace {
// FIXME: both GetParamRef and ParamReferenceReplacerRAII are good template
// candidates to be shared among LLVM / CIR codegen.

// Hunts for the parameter reference in the parameter copy/move declaration.
struct GetParamRef : public StmtVisitor<GetParamRef> {
public:
  DeclRefExpr *expr = nullptr;
  GetParamRef() {}
  void VisitDeclRefExpr(DeclRefExpr *e) {
    assert(expr == nullptr && "multilple declref in param move");
    expr = e;
  }
  void VisitStmt(Stmt *s) {
    for (Stmt *c : s->children()) {
      if (c)
        Visit(c);
    }
  }
};

// This class replaces references to parameters to their copies by changing
// the addresses in CGF.LocalDeclMap and restoring back the original values in
// its destructor.
struct ParamReferenceReplacerRAII {
  CIRGenFunction::DeclMapTy savedLocals;
  CIRGenFunction::DeclMapTy &localDeclMap;

  ParamReferenceReplacerRAII(CIRGenFunction::DeclMapTy &localDeclMap)
      : localDeclMap(localDeclMap) {}

  void addCopy(const DeclStmt *pm) {
    // Figure out what param it refers to.

    assert(pm->isSingleDecl());
    const VarDecl *vd = static_cast<const VarDecl *>(pm->getSingleDecl());
    const Expr *initExpr = vd->getInit();
    GetParamRef visitor;
    visitor.Visit(const_cast<Expr *>(initExpr));
    assert(visitor.expr);
    DeclRefExpr *dreOrig = visitor.expr;
    auto *pd = dreOrig->getDecl();

    auto it = localDeclMap.find(pd);
    assert(it != localDeclMap.end() && "parameter is not found");
    savedLocals.insert({pd, it->second});

    auto copyIt = localDeclMap.find(vd);
    assert(copyIt != localDeclMap.end() && "parameter copy is not found");
    it->second = copyIt->getSecond();
  }

  ~ParamReferenceReplacerRAII() {
    for (auto &&savedLocal : savedLocals) {
      localDeclMap.insert({savedLocal.first, savedLocal.second});
    }
  }
};
} // namespace
static void createCoroData(CIRGenFunction &cgf,
                           CIRGenFunction::CGCoroInfo &curCoro,
                           cir::CallOp coroId) {
  assert(!curCoro.data && "EmitCoroutineBodyStatement called twice?");

  curCoro.data = std::make_unique<CGCoroData>();
  curCoro.data->coroId = coroId;
}

cir::CallOp CIRGenFunction::emitCoroIDBuiltinCall(mlir::Location loc,
                                                  mlir::Value nullPtr) {
  cir::IntType int32Ty = builder.getUInt32Ty();

  const TargetInfo &ti = cgm.getASTContext().getTargetInfo();
  unsigned newAlign = ti.getNewAlign() / ti.getCharWidth();

  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroId);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(
        loc, cgm.builtinCoroId,
        cir::FuncType::get({int32Ty, voidPtrTy, voidPtrTy, voidPtrTy}, int32Ty),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(loc, fnOp,
                              mlir::ValueRange{builder.getUInt32(newAlign, loc),
                                               nullPtr, nullPtr, nullPtr});
}

cir::CallOp CIRGenFunction::emitCoroAllocBuiltinCall(mlir::Location loc) {
  cir::BoolType boolTy = builder.getBoolTy();

  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroAlloc);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(loc, cgm.builtinCoroAlloc,
                                        cir::FuncType::get({uInt32Ty}, boolTy),
                                        /*fd=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{curCoro.data->coroId.getResult()});
}

cir::CallOp
CIRGenFunction::emitCoroBeginBuiltinCall(mlir::Location loc,
                                         mlir::Value coroframeAddr) {
  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroBegin);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(
        loc, cgm.builtinCoroBegin,
        cir::FuncType::get({uInt32Ty, voidPtrTy}, voidPtrTy),
        /*fd=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(
      loc, fnOp,
      mlir::ValueRange{curCoro.data->coroId.getResult(), coroframeAddr});
}

mlir::LogicalResult
CIRGenFunction::emitCoroutineBody(const CoroutineBodyStmt &s) {
  mlir::Location openCurlyLoc = getLoc(s.getBeginLoc());
  cir::ConstantOp nullPtrCst = builder.getNullPtr(voidPtrTy, openCurlyLoc);

  auto fn = mlir::cast<cir::FuncOp>(curFn);
  fn.setCoroutine(true);
  cir::CallOp coroId = emitCoroIDBuiltinCall(openCurlyLoc, nullPtrCst);
  createCoroData(*this, curCoro, coroId);

  // Backend is allowed to elide memory allocations, to help it, emit
  // auto mem = coro.alloc() ? 0 : ... allocation code ...;
  cir::CallOp coroAlloc = emitCoroAllocBuiltinCall(openCurlyLoc);

  // Initialize address of coroutine frame to null
  CanQualType astVoidPtrTy = cgm.getASTContext().VoidPtrTy;
  mlir::Type allocaTy = convertTypeForMem(astVoidPtrTy);
  Address coroFrame =
      createTempAlloca(allocaTy, getContext().getTypeAlignInChars(astVoidPtrTy),
                       openCurlyLoc, "__coro_frame_addr",
                       /*ArraySize=*/nullptr);

  mlir::Value storeAddr = coroFrame.getPointer();
  builder.CIRBaseBuilderTy::createStore(openCurlyLoc, nullPtrCst, storeAddr);
  cir::IfOp::create(
      builder, openCurlyLoc, coroAlloc.getResult(),
      /*withElseRegion=*/false,
      /*thenBuilder=*/[&](mlir::OpBuilder &b, mlir::Location loc) {
        builder.CIRBaseBuilderTy::createStore(
            loc, emitScalarExpr(s.getAllocate()), storeAddr);
        cir::YieldOp::create(builder, loc);
      });
  curCoro.data->coroBegin =
      emitCoroBeginBuiltinCall(
          openCurlyLoc,
          cir::LoadOp::create(builder, openCurlyLoc, allocaTy, storeAddr))
          .getResult();

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (s.getReturnStmtOnAllocFailure())
    cgm.errorNYI("handle coroutine return alloc failure");

  {
    assert(!cir::MissingFeatures::generateDebugInfo());
    ParamReferenceReplacerRAII paramReplacer(localDeclMap);
    // Create mapping between parameters and copy-params for coroutine
    // function.
    llvm::ArrayRef<const Stmt *> paramMoves = s.getParamMoves();
    assert((paramMoves.size() == 0 || (paramMoves.size() == fnArgs.size())) &&
           "ParamMoves and FnArgs should be the same size for coroutine "
           "function");
    // For zipping the arg map into debug info.
    assert(!cir::MissingFeatures::generateDebugInfo());

    // Create parameter copies. We do it before creating a promise, since an
    // evolution of coroutine TS may allow promise constructor to observe
    // parameter copies.
    assert(!cir::MissingFeatures::coroOutsideFrameMD());
    for (auto *pm : paramMoves) {
      if (emitStmt(pm, /*useCurrentScope=*/true).failed())
        return mlir::failure();
      paramReplacer.addCopy(cast<DeclStmt>(pm));
    }

    if (emitStmt(s.getPromiseDeclStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    // returnValue should be valid as long as the coroutine's return type
    // is not void. The assertion could help us to reduce the check later.
    assert(returnValue.isValid() == (bool)s.getReturnStmt());
    // Now we have the promise, initialize the GRO.
    // We need to emit `get_return_object` first. According to:
    // [dcl.fct.def.coroutine]p7
    // The call to get_return_­object is sequenced before the call to
    // initial_suspend and is invoked at most once.
    //
    // So we couldn't emit return value when we emit return statment,
    // otherwise the call to get_return_object wouldn't be in front
    // of initial_suspend.
    if (returnValue.isValid())
      emitAnyExprToMem(s.getReturnValue(), returnValue,
                       s.getReturnValue()->getType().getQualifiers(),
                       /*isInit*/ true);
    assert(!cir::MissingFeatures::emitBodyAndFallthrough());
  }
  return mlir::success();
}
