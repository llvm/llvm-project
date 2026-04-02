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
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  cir::AwaitKind currentAwaitKind = cir::AwaitKind::Init;
  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  cir::CallOp coroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value coroBegin = nullptr;

  // Stores the insertion point for final suspend, this happens after the
  // promise call (return_xxx promise member) but before a cir.br to the return
  // block.
  mlir::Operation *finalSuspendInsPoint;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned coreturnCount = 0;

  // The promise type's 'unhandled_exception' handler, if it defines one.
  Stmt *exceptionHandler = nullptr;
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

RValue CIRGenFunction::emitCoroutineFrame() {
  if (curCoro.data && curCoro.data->coroBegin) {
    return RValue::get(curCoro.data->coroBegin);
  }
  cgm.errorNYI("NYI");
  return RValue();
}

static void createCoroData(CIRGenFunction &cgf,
                           CIRGenFunction::CGCoroInfo &curCoro,
                           cir::CallOp coroId) {
  assert(!curCoro.data && "EmitCoroutineBodyStatement called twice?");

  curCoro.data = std::make_unique<CGCoroData>();
  curCoro.data->coroId = coroId;
}

static mlir::LogicalResult
emitBodyAndFallthrough(CIRGenFunction &cgf, const CoroutineBodyStmt &s,
                       Stmt *body,
                       const CIRGenFunction::LexicalScope *currLexScope) {
  if (cgf.emitStmt(body, /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Note that classic codegen checks CanFallthrough by looking into the
  // availability of the insert block which is kinda brittle and unintuitive,
  // seems to be related with how landing pads are handled.
  //
  // CIRGen handles this by checking pre-existing co_returns in the current
  // scope instead.

  // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
  const bool canFallthrough = !currLexScope->hasCoreturn();
  if (canFallthrough)
    if (Stmt *onFallthrough = s.getFallthroughHandler())
      if (cgf.emitStmt(onFallthrough, /*useCurrentScope=*/true).failed())
        return mlir::failure();

  return mlir::success();
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

cir::CallOp CIRGenFunction::emitCoroEndBuiltinCall(mlir::Location loc,
                                                   mlir::Value nullPtr) {
  cir::BoolType boolTy = builder.getBoolTy();
  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroEnd);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(
        loc, cgm.builtinCoroEnd,
        cir::FuncType::get({voidPtrTy, boolTy}, boolTy),
        /*fd=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{nullPtr, builder.getBool(false, loc)});
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

    assert(!cir::MissingFeatures::ehCleanupScope());

    curCoro.data->currentAwaitKind = cir::AwaitKind::Init;
    if (emitStmt(s.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    curCoro.data->currentAwaitKind = cir::AwaitKind::User;

    // FIXME(cir): wrap emitBodyAndFallthrough with try/catch bits.
    if (s.getExceptionHandler())
      assert(!cir::MissingFeatures::coroutineExceptions());
    if (emitBodyAndFallthrough(*this, s, s.getBody(), curLexScope).failed())
      return mlir::failure();

    // Note that LLVM checks CanFallthrough by looking into the availability
    // of the insert block which is kinda brittle and unintuitive, seems to be
    // related with how landing pads are handled.
    //
    // CIRGen handles this by checking pre-existing co_returns in the current
    // scope instead.
    //
    // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
    const bool canFallthrough = curLexScope->hasCoreturn();
    const bool hasCoreturns = curCoro.data->coreturnCount > 0;
    if (canFallthrough || hasCoreturns) {
      curCoro.data->currentAwaitKind = cir::AwaitKind::Final;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(curCoro.data->finalSuspendInsPoint);
        if (emitStmt(s.getFinalSuspendStmt(), /*useCurrentScope=*/true)
                .failed())
          return mlir::failure();
      }
    }
  }
  return mlir::success();
}

static bool memberCallExpressionCanThrow(const Expr *e) {
  if (const auto *ce = dyn_cast<CXXMemberCallExpr>(e))
    if (const auto *proto =
            ce->getMethodDecl()->getType()->getAs<FunctionProtoType>())
      if (isNoexceptExceptionSpec(proto->getExceptionSpecType()) &&
          proto->canThrow() == CT_Cannot)
        return false;
  return true;
}

// Given a suspend expression which roughly looks like:
//
//   auto && x = CommonExpr();
//   if (!x.await_ready()) {
//      x.await_suspend(...); (*)
//   }
//   x.await_resume();
//
// where the result of the entire expression is the result of x.await_resume()
//
//   (*) If x.await_suspend return type is bool, it allows to veto a suspend:
//      if (x.await_suspend(...))
//        llvm_coro_suspend();
//
// This is more higher level than LLVM codegen, for that one see llvm's
// docs/Coroutines.rst for more details.
namespace {
struct LValueOrRValue {
  LValue lv;
  RValue rv;
};
} // namespace

static LValueOrRValue
emitSuspendExpression(CIRGenFunction &cgf, CGCoroData &coro,
                      CoroutineSuspendExpr const &s, cir::AwaitKind kind,
                      AggValueSlot aggSlot, bool ignoreResult,
                      mlir::Block *scopeParentBlock,
                      mlir::Value &tmpResumeRValAddr, bool forLValue) {
  [[maybe_unused]] mlir::LogicalResult awaitBuild = mlir::success();
  LValueOrRValue awaitRes;

  CIRGenFunction::OpaqueValueMapping binder =
      CIRGenFunction::OpaqueValueMapping(cgf, s.getOpaqueValue());
  CIRGenBuilderTy &builder = cgf.getBuilder();
  [[maybe_unused]] cir::AwaitOp awaitOp = cir::AwaitOp::create(
      builder, cgf.getLoc(s.getSourceRange()), kind,
      /*readyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        Expr *condExpr = s.getReadyExpr()->IgnoreParens();
        builder.createCondition(cgf.evaluateExprAsBool(condExpr));
      },
      /*suspendBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Note that differently from LLVM codegen we do not emit coro.save
        // and coro.suspend here, that should be done as part of lowering this
        // to LLVM dialect (or some other MLIR dialect)

        // A invalid suspendRet indicates "void returning await_suspend"
        mlir::Value suspendRet = cgf.emitScalarExpr(s.getSuspendExpr());

        // Veto suspension if requested by bool returning await_suspend.
        if (suspendRet) {
          cgf.cgm.errorNYI("Veto await_suspend");
        }

        // Signals the parent that execution flows to next region.
        cir::YieldOp::create(builder, loc);
      },
      /*resumeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Exception handling requires additional IR. If the 'await_resume'
        // function is marked as 'noexcept', we avoid generating this additional
        // IR.
        CXXTryStmt *tryStmt = nullptr;
        if (coro.exceptionHandler && kind == cir::AwaitKind::Init &&
            memberCallExpressionCanThrow(s.getResumeExpr()))
          cgf.cgm.errorNYI("Coro resume Exception");

        // FIXME(cir): the alloca for the resume expr should be placed in the
        // enclosing cir.scope instead.
        if (forLValue) {
          awaitRes.lv = cgf.emitLValue(s.getResumeExpr());
        } else {
          awaitRes.rv =
              cgf.emitAnyExpr(s.getResumeExpr(), aggSlot, ignoreResult);
          if (!awaitRes.rv.isIgnored()) {
            // Create the alloca in the block before the scope wrapping
            // cir.await.
            tmpResumeRValAddr = cgf.emitAlloca(
                "__coawait_resume_rval", awaitRes.rv.getValue().getType(), loc,
                CharUnits::One(),
                builder.getBestAllocaInsertPoint(scopeParentBlock));
            // Store the rvalue so we can reload it before the promise call.
            builder.CIRBaseBuilderTy::createStore(loc, awaitRes.rv.getValue(),
                                                  tmpResumeRValAddr);
          }
        }

        if (tryStmt)
          cgf.cgm.errorNYI("Coro tryStmt");

        // Returns control back to parent.
        cir::YieldOp::create(builder, loc);
      });

  assert(awaitBuild.succeeded() && "Should know how to codegen");
  return awaitRes;
}

static RValue emitSuspendExpr(CIRGenFunction &cgf,
                              const CoroutineSuspendExpr &e,
                              cir::AwaitKind kind, AggValueSlot aggSlot,
                              bool ignoreResult) {
  RValue rval;
  mlir::Location scopeLoc = cgf.getLoc(e.getSourceRange());

  // Since we model suspend / resume as an inner region, we must store
  // resume scalar results in a tmp alloca, and load it after we build the
  // suspend expression. An alternative way to do this would be to make
  // every region return a value when promise.return_value() is used, but
  // it's a bit awkward given that resume is the only region that actually
  // returns a value.
  mlir::Block *currEntryBlock = cgf.curLexScope->getEntryBlock();
  [[maybe_unused]] mlir::Value tmpResumeRValAddr;

  // No need to explicitly wrap this into a scope since the AST already uses a
  // ExprWithCleanups, which will wrap this into a cir.scope anyways.
  rval = emitSuspendExpression(cgf, *cgf.curCoro.data, e, kind, aggSlot,
                               ignoreResult, currEntryBlock, tmpResumeRValAddr,
                               /*forLValue*/ false)
             .rv;

  if (ignoreResult || rval.isIgnored())
    return rval;

  if (rval.isScalar()) {
    rval = RValue::get(cir::LoadOp::create(cgf.getBuilder(), scopeLoc,
                                           rval.getValue().getType(),
                                           tmpResumeRValAddr));
  } else if (rval.isAggregate()) {
    // This is probably already handled via AggSlot, remove this assertion
    // once we have a testcase and prove all pieces work.
    cgf.cgm.errorNYI("emitSuspendExpr Aggregate");
  } else { // complex
    cgf.cgm.errorNYI("emitSuspendExpr Complex");
  }
  return rval;
}

RValue CIRGenFunction::emitCoawaitExpr(const CoawaitExpr &e,
                                       AggValueSlot aggSlot,
                                       bool ignoreResult) {
  return emitSuspendExpr(*this, e, curCoro.data->currentAwaitKind, aggSlot,
                         ignoreResult);
}

RValue CIRGenFunction::emitCoyieldExpr(const CoyieldExpr &e,
                                       AggValueSlot aggSlot,
                                       bool ignoreResult) {
  return emitSuspendExpr(*this, e, cir::AwaitKind::Yield, aggSlot,
                         ignoreResult);
}

mlir::LogicalResult CIRGenFunction::emitCoreturnStmt(CoreturnStmt const &s) {
  ++curCoro.data->coreturnCount;
  curLexScope->setCoreturn();

  const Expr *rv = s.getOperand();
  if (rv && rv->getType()->isVoidType() && !isa<InitListExpr>(rv)) {
    // Make sure to evaluate the non initlist expression of a co_return
    // with a void expression for side effects.
    RunCleanupsScope cleanupScope(*this);
    emitIgnoredExpr(rv);
  }

  if (emitStmt(s.getPromiseCall(), /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  mlir::Location loc = getLoc(s.getSourceRange());
  mlir::Block *retBlock = curLexScope->getOrCreateRetBlock(*this, loc);
  curCoro.data->finalSuspendInsPoint =
      cir::BrOp::create(builder, loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block,
  // this will likely be an empty block.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}
