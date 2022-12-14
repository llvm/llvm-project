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
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace cir;

namespace {
enum class AwaitKind { Init, Normal, Yield, Final };
} // namespace
struct cir::CGCoroData {
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  AwaitKind CurrentAwaitKind = AwaitKind::Init;

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
                           CIRGenFunction::CGCoroInfo &CurCoro) {
  if (CurCoro.Data) {
    llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  CurCoro.Data = std::unique_ptr<CGCoroData>(new CGCoroData);
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

static mlir::LogicalResult buildBodyAndFallthrough(CIRGenFunction &CGF,
                                                   const CoroutineBodyStmt &S,
                                                   Stmt *Body) {
  if (CGF.buildStmt(Body, /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // LLVM codegen checks if a insert basic block is available in order
  // to decide whether to getFallthroughHandler, sounds like it should
  // be an assert, not clear. For CIRGen solely rely on getFallthroughHandler.
  if (Stmt *OnFallthrough = S.getFallthroughHandler())
    if (CGF.buildStmt(OnFallthrough, /*useCurrentScope=*/true).failed())
      return mlir::failure();

  return mlir::success();
}

mlir::cir::CallOp CIRGenFunction::buildCoroIDBuiltinCall(mlir::Location loc) {
  auto int8PtrTy = builder.getInt8PtrTy();
  auto int32Ty = mlir::IntegerType::get(builder.getContext(), 32);
  auto nullPtrCst = builder.create<mlir::cir::ConstantOp>(
      loc, int8PtrTy,
      mlir::cir::NullAttr::get(builder.getContext(), int8PtrTy));

  auto &TI = CGM.getASTContext().getTargetInfo();
  unsigned NewAlign = TI.getNewAlign() / TI.getCharWidth();

  mlir::Operation *builtin = CGM.getGlobalValue(builtinCoroId);
  mlir::TypeRange argTypes{int32Ty, int8PtrTy, int8PtrTy, int8PtrTy};
  mlir::TypeRange resTypes{int32Ty};

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(loc, builtinCoroId,
                                 builder.getFunctionType(argTypes, resTypes),
                                 /*FD=*/nullptr);
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  mlir::ValueRange inputArgs{builder.getInt32(NewAlign, loc), nullPtrCst,
                             nullPtrCst, nullPtrCst};
  return builder.create<mlir::cir::CallOp>(loc, fnOp, inputArgs);
}

mlir::LogicalResult
CIRGenFunction::buildCoroutineBody(const CoroutineBodyStmt &S) {
  // This is very different from LLVM codegen as the current intent is to
  // not expand too much of it here and leave it to dialect codegen.
  // In the LLVM world, this is where we create calls to coro.id,
  // coro.alloc and coro.begin.
  [[maybe_unused]] auto coroId =
      buildCoroIDBuiltinCall(getLoc(S.getBeginLoc()));
  createCoroData(*this, CurCoro);

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

    // FIXME(cir): handle promiseAddr and coro id related stuff?

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
    CurCoro.Data->CurrentAwaitKind = AwaitKind::Init;
    if (buildStmt(S.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    CurCoro.Data->CurrentAwaitKind = AwaitKind::Normal;

    // FIXME(cir): wrap buildBodyAndFallthrough with try/catch bits.
    if (S.getExceptionHandler())
      assert(!UnimplementedFeature::unhandledException() && "NYI");
    if (buildBodyAndFallthrough(*this, S, S.getBody()).failed())
      return mlir::failure();

    // See if we need to generate final suspend.
    // const bool CanFallthrough = Builder.GetInsertBlock();
    // FIXME: LLVM tracks fallthrough by checking the insertion
    // point is valid, we can probably do better.
    const bool CanFallthrough = false;
    const bool HasCoreturns = CurCoro.Data->CoreturnCount > 0;
    if (CanFallthrough || HasCoreturns) {
      CurCoro.Data->CurrentAwaitKind = AwaitKind::Final;
      if (buildStmt(S.getFinalSuspendStmt(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
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
static LValueOrRValue buildSuspendExpression(
    CIRGenFunction &CGF, CGCoroData &Coro, CoroutineSuspendExpr const &S,
    AwaitKind Kind, AggValueSlot aggSlot, bool ignoreResult, bool forLValue) {
  auto *E = S.getCommonExpr();

  auto awaitBuild = mlir::success();
  LValueOrRValue awaitRes;

  auto Binder =
      CIRGenFunction::OpaqueValueMappingData::bind(CGF, S.getOpaqueValue(), E);
  auto UnbindOnExit = llvm::make_scope_exit([&] { Binder.unbind(CGF); });
  auto &builder = CGF.getBuilder();

  [[maybe_unused]] auto awaitOp = builder.create<mlir::cir::AwaitOp>(
      CGF.getLoc(S.getSourceRange()),
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
              // `YieldOpKind::Break` tells control flow to return to parent, no
              // more regions to be executed.
              builder.create<mlir::cir::YieldOp>(loc,
                                                 mlir::cir::YieldOpKind::Break);
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
        if (Coro.ExceptionHandler && Kind == AwaitKind::Init &&
            memberCallExpressionCanThrow(S.getResumeExpr())) {
          llvm_unreachable("NYI");
        }

        // FIXME(cir): the alloca for the resume expr should be placed in the
        // enclosing cir.scope instead.
        if (forLValue)
          awaitRes.LV = CGF.buildLValue(S.getResumeExpr());
        else
          awaitRes.RV =
              CGF.buildAnyExpr(S.getResumeExpr(), aggSlot, ignoreResult);

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
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // FIXME(cir): abstract all this massive location handling elsewhere.
        SmallVector<mlir::Location, 2> locs;
        if (loc.isa<mlir::FileLineColLoc>()) {
          locs.push_back(loc);
          locs.push_back(loc);
        } else if (loc.isa<mlir::FusedLoc>()) {
          auto fusedLoc = loc.cast<mlir::FusedLoc>();
          locs.push_back(fusedLoc.getLocations()[0]);
          locs.push_back(fusedLoc.getLocations()[1]);
        }
        LexicalScopeContext lexScope{locs[0], locs[1],
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexScopeGuard{*this, &lexScope};
        rval = buildSuspendExpression(*this, *CurCoro.Data, E,
                                      CurCoro.Data->CurrentAwaitKind, aggSlot,
                                      ignoreResult, /*forLValue*/ false)
                   .RV;
      });
  return rval;
}

mlir::LogicalResult CIRGenFunction::buildCoreturnStmt(CoreturnStmt const &S) {
  ++CurCoro.Data->CoreturnCount;
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
  // FIXME: do the proper things like ReturnStmt does
  // EmitBranchThroughCleanup(CurCoro.Data->FinalJD);

  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto loc = getLoc(S.getSourceRange());
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  builder.create<mlir::cir::BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block.
  builder.createBlock(builder.getBlock()->getParent());

  // TODO(cir): LLVM codegen for a cleanup on cleanupScope here.
  return mlir::success();
}
