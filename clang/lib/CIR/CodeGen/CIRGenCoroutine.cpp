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
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace cir;

struct cir::CGCoroData {
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  mlir::cir::AwaitKind CurrentAwaitKind = mlir::cir::AwaitKind::init;

  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  mlir::cir::CallOp CoroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value CoroBegin = nullptr;

  // Stores the insertion point for final suspend, this happens after the
  // promise call (return_xxx promise member) but before a cir.br to the return
  // block.
  mlir::Operation *FinalSuspendInsPoint;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned CoreturnCount = 0;

  // The promise type's 'unhandled_exception' handler, if it defines one.
  Stmt *ExceptionHandler = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
CIRGenFunction::CGCoroInfo::CGCoroInfo() {}
CIRGenFunction::CGCoroInfo::~CGCoroInfo() {}

static void createCoroData(CIRGenFunction &CGF,
                           CIRGenFunction::CGCoroInfo &CurCoro,
                           mlir::cir::CallOp CoroId) {
  if (CurCoro.Data) {
    llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  CurCoro.Data = std::unique_ptr<CGCoroData>(new CGCoroData);
  CurCoro.Data->CoroId = CoroId;
}

namespace {
// FIXME: both GetParamRef and ParamReferenceReplacerRAII are good template
// candidates to be shared among LLVM / CIR codegen.

// Hunts for the parameter reference in the parameter copy/move declaration.
struct GetParamRef : public StmtVisitor<GetParamRef> {
public:
  DeclRefExpr *Expr = nullptr;
  GetParamRef() {}
  void VisitDeclRefExpr(DeclRefExpr *E) {
    assert(Expr == nullptr && "multilple declref in param move");
    Expr = E;
  }
  void VisitStmt(Stmt *S) {
    for (auto *C : S->children()) {
      if (C)
        Visit(C);
    }
  }
};

// This class replaces references to parameters to their copies by changing
// the addresses in CGF.LocalDeclMap and restoring back the original values in
// its destructor.
struct ParamReferenceReplacerRAII {
  CIRGenFunction::DeclMapTy SavedLocals;
  CIRGenFunction::DeclMapTy &LocalDeclMap;

  ParamReferenceReplacerRAII(CIRGenFunction::DeclMapTy &LocalDeclMap)
      : LocalDeclMap(LocalDeclMap) {}

  void addCopy(DeclStmt const *PM) {
    // Figure out what param it refers to.

    assert(PM->isSingleDecl());
    VarDecl const *VD = static_cast<VarDecl const *>(PM->getSingleDecl());
    Expr const *InitExpr = VD->getInit();
    GetParamRef Visitor;
    Visitor.Visit(const_cast<Expr *>(InitExpr));
    assert(Visitor.Expr);
    DeclRefExpr *DREOrig = Visitor.Expr;
    auto *PD = DREOrig->getDecl();

    auto it = LocalDeclMap.find(PD);
    assert(it != LocalDeclMap.end() && "parameter is not found");
    SavedLocals.insert({PD, it->second});

    auto copyIt = LocalDeclMap.find(VD);
    assert(copyIt != LocalDeclMap.end() && "parameter copy is not found");
    it->second = copyIt->getSecond();
  }

  ~ParamReferenceReplacerRAII() {
    for (auto &&SavedLocal : SavedLocals) {
      LocalDeclMap.insert({SavedLocal.first, SavedLocal.second});
    }
  }
};
} // namespace

// Emit coroutine intrinsic and patch up arguments of the token type.
RValue CIRGenFunction::buildCoroutineIntrinsic(const CallExpr *E,
                                               unsigned int IID) {
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::buildCoroutineFrame() {
  if (CurCoro.Data && CurCoro.Data->CoroBegin) {
    return RValue::get(CurCoro.Data->CoroBegin);
  }
  llvm_unreachable("NYI");
}

static mlir::LogicalResult
buildBodyAndFallthrough(CIRGenFunction &CGF, const CoroutineBodyStmt &S,
                        Stmt *Body,
                        const CIRGenFunction::LexicalScope *currLexScope) {
  if (CGF.buildStmt(Body, /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Note that LLVM checks CanFallthrough by looking into the availability
  // of the insert block which is kinda brittle and unintuitive, seems to be
  // related with how landing pads are handled.
  //
  // CIRGen handles this by checking pre-existing co_returns in the current
  // scope instead. Are we missing anything?
  //
  // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
  const bool CanFallthrough = !currLexScope->hasCoreturn();
  if (CanFallthrough)
    if (Stmt *OnFallthrough = S.getFallthroughHandler())
      if (CGF.buildStmt(OnFallthrough, /*useCurrentScope=*/true).failed())
        return mlir::failure();

  return mlir::success();
}

mlir::cir::CallOp CIRGenFunction::buildCoroIDBuiltinCall(mlir::Location loc,
                                                         mlir::Value nullPtr) {
  auto int32Ty = builder.getUInt32Ty();

  auto &TI = CGM.getASTContext().getTargetInfo();
  unsigned NewAlign = TI.getNewAlign() / TI.getCharWidth();

  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroId);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroId,
        mlir::cir::FuncType::get({int32Ty, VoidPtrTy, VoidPtrTy, VoidPtrTy},
                                 int32Ty),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.create<mlir::cir::CallOp>(
      loc, fnOp,
      mlir::ValueRange{builder.getUInt32(NewAlign, loc), nullPtr, nullPtr,
                       nullPtr});
}

mlir::cir::CallOp
CIRGenFunction::buildCoroAllocBuiltinCall(mlir::Location loc) {
  auto boolTy = builder.getBoolTy();
  auto int32Ty = builder.getUInt32Ty();

  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroAlloc);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroAlloc,
        mlir::cir::FuncType::get({int32Ty}, boolTy),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.create<mlir::cir::CallOp>(
      loc, fnOp, mlir::ValueRange{CurCoro.Data->CoroId.getResult(0)});
}

mlir::cir::CallOp
CIRGenFunction::buildCoroBeginBuiltinCall(mlir::Location loc,
                                          mlir::Value coroframeAddr) {
  auto int32Ty = builder.getUInt32Ty();
  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroBegin);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroBegin,
        mlir::cir::FuncType::get({int32Ty, VoidPtrTy},
                                 VoidPtrTy),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.create<mlir::cir::CallOp>(
      loc, fnOp,
      mlir::ValueRange{CurCoro.Data->CoroId.getResult(0), coroframeAddr});
}

mlir::cir::CallOp CIRGenFunction::buildCoroEndBuiltinCall(mlir::Location loc,
                                                          mlir::Value nullPtr) {
  auto boolTy = builder.getBoolTy();
  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroEnd);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroEnd,
        mlir::cir::FuncType::get({VoidPtrTy, boolTy}, boolTy),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.create<mlir::cir::CallOp>(
      loc, fnOp, mlir::ValueRange{nullPtr, builder.getBool(false, loc)});
}

mlir::LogicalResult
CIRGenFunction::buildCoroutineBody(const CoroutineBodyStmt &S) {
  auto openCurlyLoc = getLoc(S.getBeginLoc());
  auto nullPtrCst = builder.getNullPtr(VoidPtrTy, openCurlyLoc);

  auto Fn = dyn_cast<mlir::cir::FuncOp>(CurFn);
  assert(Fn && "other callables NYI");
  Fn.setCoroutineAttr(mlir::UnitAttr::get(builder.getContext()));
  auto coroId = buildCoroIDBuiltinCall(openCurlyLoc, nullPtrCst);
  createCoroData(*this, CurCoro, coroId);

  // Backend is allowed to elide memory allocations, to help it, emit
  // auto mem = coro.alloc() ? 0 : ... allocation code ...;
  auto coroAlloc = buildCoroAllocBuiltinCall(openCurlyLoc);

  // Initialize address of coroutine frame to null
  auto astVoidPtrTy = CGM.getASTContext().VoidPtrTy;
  auto allocaTy = getTypes().convertTypeForMem(astVoidPtrTy);
  Address coroFrame =
      CreateTempAlloca(allocaTy, getContext().getTypeAlignInChars(astVoidPtrTy),
                       openCurlyLoc, "__coro_frame_addr",
                       /*ArraySize=*/nullptr);

  auto storeAddr = coroFrame.getPointer();
  builder.create<mlir::cir::StoreOp>(openCurlyLoc, nullPtrCst, storeAddr);
  builder.create<mlir::cir::IfOp>(openCurlyLoc, coroAlloc.getResult(0),
                                  /*withElseRegion=*/false,
                                  /*thenBuilder=*/
                                  [&](mlir::OpBuilder &b, mlir::Location loc) {
                                    builder.create<mlir::cir::StoreOp>(
                                        loc, buildScalarExpr(S.getAllocate()),
                                        storeAddr);
                                    builder.create<mlir::cir::YieldOp>(loc);
                                  });

  CurCoro.Data->CoroBegin =
      buildCoroBeginBuiltinCall(
          openCurlyLoc,
          builder.create<mlir::cir::LoadOp>(openCurlyLoc, allocaTy, storeAddr))
          .getResult(0);

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (auto *RetOnAllocFailure = S.getReturnStmtOnAllocFailure())
    llvm_unreachable("NYI");

  {
    // FIXME(cir): create a new scope to copy out the params?
    // LLVM create scope cleanups here, but might be due to the use
    // of many basic blocks?
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
    ParamReferenceReplacerRAII ParamReplacer(LocalDeclMap);

    // Create mapping between parameters and copy-params for coroutine
    // function.
    llvm::ArrayRef<const Stmt *> ParamMoves = S.getParamMoves();
    assert((ParamMoves.size() == 0 || (ParamMoves.size() == FnArgs.size())) &&
           "ParamMoves and FnArgs should be the same size for coroutine "
           "function");
    // For zipping the arg map into debug info.
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");

    // Create parameter copies. We do it before creating a promise, since an
    // evolution of coroutine TS may allow promise constructor to observe
    // parameter copies.
    for (auto *PM : S.getParamMoves()) {
      if (buildStmt(PM, /*useCurrentScope=*/true).failed())
        return mlir::failure();
      ParamReplacer.addCopy(cast<DeclStmt>(PM));
    }

    if (buildStmt(S.getPromiseDeclStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    // ReturnValue should be valid as long as the coroutine's return type
    // is not void. The assertion could help us to reduce the check later.
    assert(ReturnValue.isValid() == (bool)S.getReturnStmt());
    // Now we have the promise, initialize the GRO.
    // We need to emit `get_return_object` first. According to:
    // [dcl.fct.def.coroutine]p7
    // The call to get_return_­object is sequenced before the call to
    // initial_suspend and is invoked at most once.
    //
    // So we couldn't emit return value when we emit return statment,
    // otherwise the call to get_return_object wouldn't be in front
    // of initial_suspend.
    if (ReturnValue.isValid()) {
      buildAnyExprToMem(S.getReturnValue(), ReturnValue,
                        S.getReturnValue()->getType().getQualifiers(),
                        /*IsInit*/ true);
    }

    // FIXME(cir): EHStack.pushCleanup<CallCoroEnd>(EHCleanup);
    CurCoro.Data->CurrentAwaitKind = mlir::cir::AwaitKind::init;
    if (buildStmt(S.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    CurCoro.Data->CurrentAwaitKind = mlir::cir::AwaitKind::user;

    // FIXME(cir): wrap buildBodyAndFallthrough with try/catch bits.
    if (S.getExceptionHandler())
      assert(!UnimplementedFeature::unhandledException() && "NYI");
    if (buildBodyAndFallthrough(*this, S, S.getBody(), currLexScope).failed())
      return mlir::failure();

    // Note that LLVM checks CanFallthrough by looking into the availability
    // of the insert block which is kinda brittle and unintuitive, seems to be
    // related with how landing pads are handled.
    //
    // CIRGen handles this by checking pre-existing co_returns in the current
    // scope instead. Are we missing anything?
    //
    // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
    const bool CanFallthrough = currLexScope->hasCoreturn();
    const bool HasCoreturns = CurCoro.Data->CoreturnCount > 0;
    if (CanFallthrough || HasCoreturns) {
      CurCoro.Data->CurrentAwaitKind = mlir::cir::AwaitKind::final;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(CurCoro.Data->FinalSuspendInsPoint);
        if (buildStmt(S.getFinalSuspendStmt(), /*useCurrentScope=*/true)
                .failed())
          return mlir::failure();
      }
    }
  }
  return mlir::success();
}

static bool memberCallExpressionCanThrow(const Expr *E) {
  if (const auto *CE = dyn_cast<CXXMemberCallExpr>(E))
    if (const auto *Proto =
            CE->getMethodDecl()->getType()->getAs<FunctionProtoType>())
      if (isNoexceptExceptionSpec(Proto->getExceptionSpecType()) &&
          Proto->canThrow() == CT_Cannot)
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
  LValue LV;
  RValue RV;
};
} // namespace
static LValueOrRValue
buildSuspendExpression(CIRGenFunction &CGF, CGCoroData &Coro,
                       CoroutineSuspendExpr const &S, mlir::cir::AwaitKind Kind,
                       AggValueSlot aggSlot, bool ignoreResult,
                       mlir::Block *scopeParentBlock,
                       mlir::Value &tmpResumeRValAddr, bool forLValue) {
  auto *E = S.getCommonExpr();

  auto awaitBuild = mlir::success();
  LValueOrRValue awaitRes;

  auto Binder =
      CIRGenFunction::OpaqueValueMappingData::bind(CGF, S.getOpaqueValue(), E);
  auto UnbindOnExit = llvm::make_scope_exit([&] { Binder.unbind(CGF); });
  auto &builder = CGF.getBuilder();

  [[maybe_unused]] auto awaitOp = builder.create<mlir::cir::AwaitOp>(
      CGF.getLoc(S.getSourceRange()), Kind,
      /*readyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto *cond = S.getReadyExpr();
        cond = cond->IgnoreParens();
        mlir::Value condV = CGF.evaluateExprAsBool(cond);

        builder.create<mlir::cir::IfOp>(
            loc, condV, /*withElseRegion=*/false,
            /*thenBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              // If expression is ready, no need to suspend,
              // `YieldOpKind::NoSuspend` tells control flow to return to
              // parent, no more regions to be executed.
              builder.create<mlir::cir::YieldOp>(
                  loc, mlir::cir::YieldOpKind::NoSuspend);
            });

        if (!condV) {
          awaitBuild = mlir::failure();
          return;
        }

        // Signals the parent that execution flows to next region.
        builder.create<mlir::cir::YieldOp>(loc);
      },
      /*suspendBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Note that differently from LLVM codegen we do not emit coro.save
        // and coro.suspend here, that should be done as part of lowering this
        // to LLVM dialect (or some other MLIR dialect)

        // A invalid suspendRet indicates "void returning await_suspend"
        auto suspendRet = CGF.buildScalarExpr(S.getSuspendExpr());

        // Veto suspension if requested by bool returning await_suspend.
        if (suspendRet) {
          // From LLVM codegen:
          // if (SuspendRet != nullptr && SuspendRet->getType()->isIntegerTy(1))
          llvm_unreachable("NYI");
        }

        // Signals the parent that execution flows to next region.
        builder.create<mlir::cir::YieldOp>(loc);
      },
      /*resumeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Exception handling requires additional IR. If the 'await_resume'
        // function is marked as 'noexcept', we avoid generating this additional
        // IR.
        CXXTryStmt *TryStmt = nullptr;
        if (Coro.ExceptionHandler && Kind == mlir::cir::AwaitKind::init &&
            memberCallExpressionCanThrow(S.getResumeExpr())) {
          llvm_unreachable("NYI");
        }

        // FIXME(cir): the alloca for the resume expr should be placed in the
        // enclosing cir.scope instead.
        if (forLValue)
          awaitRes.LV = CGF.buildLValue(S.getResumeExpr());
        else {
          awaitRes.RV =
              CGF.buildAnyExpr(S.getResumeExpr(), aggSlot, ignoreResult);
          if (!awaitRes.RV.isIgnored()) {
            // Create the alloca in the block before the scope wrapping
            // cir.await.
            tmpResumeRValAddr = CGF.buildAlloca(
                "__coawait_resume_rval", awaitRes.RV.getScalarVal().getType(),
                loc, CharUnits::One(),
                builder.getBestAllocaInsertPoint(scopeParentBlock));
            // Store the rvalue so we can reload it before the promise call.
            builder.create<mlir::cir::StoreOp>(loc, awaitRes.RV.getScalarVal(),
                                               tmpResumeRValAddr);
          }
        }

        if (TryStmt) {
          llvm_unreachable("NYI");
        }

        // Returns control back to parent.
        builder.create<mlir::cir::YieldOp>(loc);
      });

  assert(awaitBuild.succeeded() && "Should know how to codegen");
  return awaitRes;
}

RValue CIRGenFunction::buildCoawaitExpr(const CoawaitExpr &E,
                                        AggValueSlot aggSlot,
                                        bool ignoreResult) {
  RValue rval;
  auto scopeLoc = getLoc(E.getSourceRange());

  // Since we model suspend / resume as an inner region, we must store
  // resume scalar results in a tmp alloca, and load it after we build the
  // suspend expression. An alternative way to do this would be to make
  // every region return a value when promise.return_value() is used, but
  // it's a bit awkward given that resume is the only region that actually
  // returns a value.
  mlir::Block *currEntryBlock = currLexScope->getEntryBlock();
  [[maybe_unused]] mlir::Value tmpResumeRValAddr;

  // No need to explicitly wrap this into a scope since the AST already uses a
  // ExprWithCleanups, which will wrap this into a cir.scope anyways.
  rval = buildSuspendExpression(*this, *CurCoro.Data, E,
                                CurCoro.Data->CurrentAwaitKind, aggSlot,
                                ignoreResult, currEntryBlock, tmpResumeRValAddr,
                                /*forLValue*/ false)
             .RV;

  if (ignoreResult || rval.isIgnored())
    return rval;

  if (rval.isScalar()) {
    rval = RValue::get(builder.create<mlir::cir::LoadOp>(
        scopeLoc, rval.getScalarVal().getType(), tmpResumeRValAddr));
  } else if (rval.isAggregate()) {
    // This is probably already handled via AggSlot, remove this assertion
    // once we have a testcase and prove all pieces work.
    llvm_unreachable("NYI");
  } else { // complex
    llvm_unreachable("NYI");
  }
  return rval;
}

mlir::LogicalResult CIRGenFunction::buildCoreturnStmt(CoreturnStmt const &S) {
  ++CurCoro.Data->CoreturnCount;
  currLexScope->setCoreturn();

  const Expr *RV = S.getOperand();
  if (RV && RV->getType()->isVoidType() && !isa<InitListExpr>(RV)) {
    // Make sure to evaluate the non initlist expression of a co_return
    // with a void expression for side effects.
    // FIXME(cir): add scope
    // RunCleanupsScope cleanupScope(*this);
    buildIgnoredExpr(RV);
  }
  if (buildStmt(S.getPromiseCall(), /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto loc = getLoc(S.getSourceRange());
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  CurCoro.Data->FinalSuspendInsPoint =
      builder.create<mlir::cir::BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block,
  // this will likely be an empty block.
  builder.createBlock(builder.getBlock()->getParent());

  // TODO(cir): LLVM codegen for a cleanup on cleanupScope here.
  return mlir::success();
}
