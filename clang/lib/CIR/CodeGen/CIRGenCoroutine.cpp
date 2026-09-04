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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
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
  cir::CoroIdOp coroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value coroBegin = nullptr;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned coreturnCount = 0;

  // The promise type's 'unhandled_exception' handler, if it defines one.
  Stmt *exceptionHandler = nullptr;

  // Stores the last emitted coro.free for the deallocate expressions, we use it
  // to wrap dealloc code with if(auto mem = coro.free) dealloc(mem).
  cir::CoroFreeOp lastCoroFree = nullptr;

  // A temporary bool alloca that stores whether 'await_resume' threw an
  // exception. If it did, 'true' is stored in this variable, and the coroutine
  // body must be skipped. If the promise type does not define an exception
  // handler, this is null.
  Address resumeEHVar = Address::invalid();

  // If coro.id came from the builtin, remember the expression to give better
  // diagnostic. If CoroIdExpr is nullptr, the coro.id was created by
  // EmitCoroutineBody.
  CallExpr const *coroIdExpr = nullptr;
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

namespace {
// Make sure to call coro.delete on scope exit.
struct CallCoroDelete final : public EHScopeStack::Cleanup {
  Stmt *deallocate;

  // Emit "if (coro.free(CoroId, CoroBegin)) Deallocate;"

  // Note: That deallocation will be emitted twice: once for a normal exit and
  // once for exceptional exit. This usage is safe because Deallocate does not
  // contain any declarations. The SubStmtBuilder::makeNewAndDeleteExpr()
  // builds a single call to a deallocation function which is safe to emit
  // multiple times.
  void emit(CIRGenFunction &cgf, Flags) override {
    // Remember the current point, as we are going to emit deallocation code
    // first to get to coro.free instruction that is an argument to a delete
    // call.

    if (cgf.emitStmt(deallocate, /*useCurrentScope=*/true).failed()) {
      cgf.cgm.error(deallocate->getBeginLoc(),
                    "failed to emit coroutine deallocation expression");
      return;
    }

    CIRGenBuilderTy &builder = cgf.getBuilder();
    cir::CoroFreeOp coroFree = cgf.curCoro.data->lastCoroFree;

    if (!coroFree) {
      cgf.cgm.error(deallocate->getBeginLoc(),
                    "Deallocation expression does not refer to coro.free");
      return;
    }

    builder.setInsertionPointAfter(coroFree);
    mlir::Value isPtrNotNull = builder.createPtrIsNotNull(coroFree.getResult());

    llvm::SmallVector<mlir::Operation *> opsToMove;
    mlir::Block *block = builder.getInsertionBlock();
    mlir::Block::iterator it(isPtrNotNull.getDefiningOp());

    for (++it; it != block->end(); ++it)
      opsToMove.push_back(&*it);

    auto ifOp =
        cir::IfOp::create(builder, cgf.getLoc(deallocate->getSourceRange()),
                          isPtrNotNull, /*withElseRegion*/ false,
                          [&](mlir::OpBuilder &builder, mlir::Location loc) {
                            cir::YieldOp::create(builder, loc);
                          });

    mlir::Operation *yieldOp = ifOp.getThenRegion().back().getTerminator();
    for (auto *op : opsToMove)
      op->moveBefore(yieldOp);
  }
  explicit CallCoroDelete(Stmt *deallocStmt) : deallocate(deallocStmt) {}
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
                           cir::CoroIdOp coroId,
                           CallExpr const *coroIdExpr = nullptr) {

  if (curCoro.data) {
    if (curCoro.data->coroIdExpr)
      cgf.cgm.error(coroIdExpr->getBeginLoc(),
                    "only one __builtin_coro_id can be used in a function");
    else if (coroIdExpr)
      cgf.cgm.error(coroIdExpr->getBeginLoc(),
                    "__builtin_coro_id shall not be used in a C++ coroutine");
    else
      llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  curCoro.data = std::make_unique<CGCoroData>();
  curCoro.data->coroId = coroId;
  curCoro.data->coroIdExpr = coroIdExpr;
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

cir::CoroIdOp CIRGenFunction::emitCoroIDBuiltinCall(const CallExpr *e) {
  mlir::Location loc = getLoc(e->getBeginLoc());

  llvm::SmallVector<mlir::Value, 4> args;
  for (const Expr *arg : e->arguments())
    args.push_back(emitScalarExpr(arg));

  auto coroId = cir::CoroIdOp::create(cgm.getBuilder(), loc, args);
  createCoroData(*this, curCoro, coroId, e);
  return coroId;
}

cir::CoroAllocOp CIRGenFunction::emitCoroAllocBuiltinCall(const CallExpr *e) {
  mlir::Location loc = getLoc(e->getBeginLoc());
  if (!curCoro.data || !curCoro.data->coroId) {
    cgm.error(e->getBeginLoc(), "this builtin expect that __builtin_coro_id has"
                                " been used earlier in this function");
    return {};
  }

  return cir::CoroAllocOp::create(
      cgm.getBuilder(), loc,
      mlir::ValueRange{curCoro.data->coroId.getResult()});
}

cir::CoroBeginOp CIRGenFunction::emitCoroBeginBuiltinCall(const CallExpr *e) {

  mlir::Location loc = getLoc(e->getBeginLoc());
  if (!curCoro.data || !curCoro.data->coroId) {
    cgm.error(e->getBeginLoc(), "this builtin expect that __builtin_coro_id has"
                                " been used earlier in this function");
    return {};
  }
  llvm::SmallVector<mlir::Value, 2> args;
  args.push_back(curCoro.data->coroId.getResult());
  for (const Expr *arg : e->arguments())
    args.push_back(emitScalarExpr(arg));

  auto coroBegin = cir::CoroBeginOp::create(cgm.getBuilder(), loc, args);
  curCoro.data->coroBegin = coroBegin;
  return coroBegin;
}

cir::CoroEndOp CIRGenFunction::emitCoroEndBuiltinCall(const CallExpr *e) {

  mlir::Location loc = getLoc(e->getBeginLoc());
  CIRGenBuilderTy &builder = cgm.getBuilder();
  llvm::SmallVector<mlir::Value, 3> args;
  for (const Expr *arg : e->arguments())
    args.push_back(emitScalarExpr(arg));
  args.push_back(cir::TokenNoneOp::create(builder, loc));
  return cir::CoroEndOp::create(builder, loc, {cgm.voidTy}, args);
}

cir::CoroFreeOp CIRGenFunction::emitCoroFreeBuiltin(const CallExpr *e) {
  mlir::Location loc = getLoc(e->getBeginLoc());

  if (!curCoro.data || !curCoro.data->coroId) {
    cgm.error(e->getBeginLoc(), "this builtin expect that __builtin_coro_id has"
                                " been used earlier in this function");
    return {};
  }

  auto coroFree =
      cir::CoroFreeOp::create(cgm.getBuilder(), loc,
                              mlir::ValueRange{curCoro.data->coroId.getResult(),
                                               curCoro.data->coroBegin});

  curCoro.data->lastCoroFree = coroFree;
  return coroFree;
}

cir::CoroSizeOp CIRGenFunction::emitCoroSizeBuiltinCall(const CallExpr *e) {
  mlir::Location loc = getLoc(e->getBeginLoc());
  return cir::CoroSizeOp::create(cgm.getBuilder(), loc);
}

static mlir::LogicalResult
coroutineBodyExceptionHelper(CIRGenFunction &cgf, const CoroutineBodyStmt &s) {

  CXXCatchStmt catchStmt(s.getBeginLoc(), /*exDecl=*/nullptr,
                         cgf.curCoro.data->exceptionHandler);
  auto *tryStmt = CXXTryStmt::Create(cgf.getContext(), s.getBeginLoc(),
                                     s.getBody(), &catchStmt);
  struct handlerEmitter final : CIRGenFunction::cxxTryBodyEmitter {
    const CoroutineBodyStmt &s;

    handlerEmitter(const CoroutineBodyStmt &s) : s(s) /*, scope(scope)*/ {}
    mlir::LogicalResult operator()(CIRGenFunction &cgf) override {
      return emitBodyAndFallthrough(cgf, s, s.getBody(), cgf.curLexScope);
    }
    ~handlerEmitter() override = default;
  } emitter{s};

  mlir::LogicalResult res = cgf.emitCXXTryStmt(*tryStmt, emitter);

  return res;
}

mlir::LogicalResult
CIRGenFunction::emitCoroutineBody(const CoroutineBodyStmt &s) {
  mlir::Location openCurlyLoc = getLoc(s.getBeginLoc());
  cir::ConstantOp nullPtrCst = builder.getNullPtr(voidPtrTy, openCurlyLoc);

  auto fn = mlir::cast<cir::FuncOp>(curFn);
  fn.setCoroutine(true);
  const TargetInfo &ti = cgm.getASTContext().getTargetInfo();
  unsigned newAlign = ti.getNewAlign() / ti.getCharWidth();

  cir::CoroIdOp coroId = cir::CoroIdOp::create(
      cgm.getBuilder(), openCurlyLoc,
      mlir::ValueRange{builder.getUInt32(newAlign, openCurlyLoc), nullPtrCst,
                       nullPtrCst, nullPtrCst});
  createCoroData(*this, curCoro, coroId);

  // Backend is allowed to elide memory allocations, to help it, emit
  // auto mem = coro.alloc() ? 0 : ... allocation code ...;
  cir::CoroAllocOp coroAlloc = cir::CoroAllocOp::create(
      cgm.getBuilder(), openCurlyLoc,
      mlir::ValueRange{curCoro.data->coroId.getResult()});

  // Initialize address of coroutine frame to null
  CanQualType astVoidPtrTy = cgm.getASTContext().VoidPtrTy;
  mlir::Type allocaTy = convertTypeForMem(astVoidPtrTy);
  Address coroFrame =
      createTempAlloca(allocaTy, getContext().getTypeAlignInChars(astVoidPtrTy),
                       openCurlyLoc, "__coro_frame_addr",
                       /*ArraySize=*/nullptr);

  mlir::Value storeAddr = coroFrame.getPointer();
  builder.CIRBaseBuilderTy::createStore(openCurlyLoc, nullPtrCst, storeAddr);
  mlir::LogicalResult res = mlir::success();
  cir::IfOp::create(
      builder, openCurlyLoc, coroAlloc.getResult(),
      /*withElseRegion=*/false,
      /*thenBuilder=*/[&](mlir::OpBuilder &b, mlir::Location loc) {
        mlir::Value allocatedPtr = emitScalarExpr(s.getAllocate());
        builder.CIRBaseBuilderTy::createStore(loc, allocatedPtr, storeAddr);
        // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
        if (Stmt *retOnAllocFailure = s.getReturnStmtOnAllocFailure()) {
          mlir::Value isPtrNull = builder.createPtrIsNull(allocatedPtr);
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          cir::IfOp::create(builder, loc, isPtrNull, /*withElseRegion=*/false,
                            [&](mlir::OpBuilder &b, mlir::Location loc) {
                              res = emitStmt(retOnAllocFailure,
                                             /*useCurrentScope=*/true);
                              cir::UnreachableOp::create(builder, loc);
                            });
        }
        cir::YieldOp::create(builder, loc);
      });

  if (res.failed())
    return res;

  curCoro.data->coroBegin = cir::CoroBeginOp::create(
      cgm.getBuilder(), openCurlyLoc,
      mlir::ValueRange{
          curCoro.data->coroId.getResult(),
          cir::LoadOp::create(builder, openCurlyLoc, allocaTy, storeAddr)});

  {
    assert(!cir::MissingFeatures::generateDebugInfo());
    ParamReferenceReplacerRAII paramReplacer(localDeclMap);
    RunCleanupsScope resumeScope(*this);
    ehStack.pushCleanup<CallCoroDelete>(NormalAndEHCleanup, s.getDeallocate());
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
    curCoro.data->exceptionHandler = s.getExceptionHandler();

    if (emitStmt(s.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    curCoro.data->currentAwaitKind = cir::AwaitKind::User;

    mlir::OpBuilder::InsertPoint userBody;
    auto coroBodyOp =
        cir::CoroBodyOp::create(builder, openCurlyLoc, /*scopeBuilder=*/
                                [&](mlir::OpBuilder &b, mlir::Location loc) {
                                  userBody = b.saveInsertionPoint();
                                });
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(userBody);
      if (curCoro.data->exceptionHandler) {
        // This bit of code is supposed to do:
        //
        // if (await-resume-didnt-throw-exception) {
        //   try {
        //     coroutine-body
        //   } catch (...) {
        //     unhandled_exception();
        //   }
        // }
        //
        // IF resume couldn't have thrown an exception(await_resume is
        // noexcept), we skip the 'if'.
        //
        // Note that we've reversed the condition of the 'if' from classic
        // codegen so that we don't need an 'else' block.
        if (curCoro.data->resumeEHVar.isValid()) {
          mlir::Value shouldSkip = builder.createFlagLoad(
              openCurlyLoc, curCoro.data->resumeEHVar.getPointer());
          mlir::LogicalResult res = mlir::success();
          cir::IfOp::create(builder, openCurlyLoc, shouldSkip,
                            /*withElseRegion=*/false,
                            [&](mlir::OpBuilder &b, mlir::Location loc) {
                              res = coroutineBodyExceptionHelper(*this, s);
                              builder.createYield(openCurlyLoc);
                            });

          if (res.failed())
            return mlir::failure();

        } else if (coroutineBodyExceptionHelper(*this, s).failed()) {
          return mlir::failure();
        }
      } else if (emitBodyAndFallthrough(*this, s, s.getBody(), curLexScope)
                     .failed()) {
        return mlir::failure();
      }
    }

    mlir::Block &coroBodyBlock = coroBodyOp.getBody().back();
    if (!coroBodyBlock.mightHaveTerminator()) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&coroBodyBlock);
      cir::YieldOp::create(builder, openCurlyLoc);
    }

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
        if (emitStmt(s.getFinalSuspendStmt(), /*useCurrentScope=*/true)
                .failed())
          return mlir::failure();
      }
    }
  }

  cir::ConstantOp nullHandler =
      builder.getNullPtr(builder.getVoidPtrTy(), openCurlyLoc);
  cir::ConstantOp noUnwind = builder.getBool(false, openCurlyLoc);
  auto tkNone = cir::TokenNoneOp::create(builder, openCurlyLoc);
  cir::CoroEndOp::create(builder, openCurlyLoc, nullHandler, noUnwind, tkNone);

  if (auto *ret = cast_or_null<ReturnStmt>(s.getReturnStmt())) {
    // Since we already emitted the return value above, so we shouldn't
    // emit it again here.
    Expr *previousRetValue = ret->getRetValue();
    ret->setRetValue(nullptr);
    if (emitStmt(ret, /*useCurrentScope=*/true).failed())
      return mlir::failure();
    // Set the return value back. The code generator, as the AST **Consumer**,
    // shouldn't change the AST.
    ret->setRetValue(previousRetValue);
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
// docs/Coroutines.md for more details.
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
        if (coro.exceptionHandler && kind == cir::AwaitKind::Init &&
            memberCallExpressionCanThrow(s.getResumeExpr())) {
          // we are basically just emitting:
          // resumeEh = false;
          // try {
          //   resumeExpr();
          //   resumeEh = true;
          // } catch(...) {
          //   exceptionHandler();
          // }
          // Note the values of resumeEh are reversed from classic codegen,
          // simply so we can use an 'IfOp' without a 'else' later.
          ASTContext &ctx = cgf.getContext();
          SourceLocation resumeLoc = s.getResumeExpr()->getExprLoc();
          mlir::Location mlirLoc = cgf.getLoc(resumeLoc);
          coro.resumeEHVar = cgf.createTempAlloca(
              builder.getBoolTy(), ctx.getTypeAlignInChars(ctx.BoolTy), mlirLoc,
              "resume.eh");
          builder.createFlagStore(mlirLoc, false,
                                  coro.resumeEHVar.getPointer());

          CXXCatchStmt catchStmt(resumeLoc,
                                 /*exDecl=*/nullptr, coro.exceptionHandler);
          auto *tryBody =
              CompoundStmt::Create(ctx, s.getResumeExpr(), FPOptionsOverride(),
                                   resumeLoc, resumeLoc);
          CXXTryStmt *tryStmt =
              CXXTryStmt::Create(ctx, resumeLoc, tryBody, &catchStmt);

          struct resumeEmitter final : CIRGenFunction::cxxTryBodyEmitter {
            const CXXTryStmt &tryStmt;
            mlir::Location loc;
            mlir::Value resumeEHVar;
            resumeEmitter(const CXXTryStmt &tryStmt, mlir::Location loc,
                          Address resumeEHVar)
                : tryStmt(tryStmt), loc(loc),
                  resumeEHVar(resumeEHVar.getPointer()) {}

            mlir::LogicalResult operator()(CIRGenFunction &cgf) override {
              mlir::LogicalResult res =
                  cgf.emitStmt(tryStmt.getTryBlock(), /*useCurrentScope=*/true);
              cgf.getBuilder().createFlagStore(loc, true, resumeEHVar);
              return res;
            }

            ~resumeEmitter() override = default;
          } emitter{*tryStmt, mlirLoc, coro.resumeEHVar};

          awaitBuild = cgf.emitCXXTryStmt(*tryStmt, emitter);
          // We are not supposed to obtain the value from init suspend
          // await_resume().
          awaitRes.rv = RValue::getIgnored();
        } else if (forLValue) {
          // FIXME(cir): the alloca for the resume expr should be placed in the
          // enclosing cir.scope instead.
          awaitRes.lv = cgf.emitLValue(s.getResumeExpr());
        } else {
          awaitRes.rv =
              cgf.emitAnyExpr(s.getResumeExpr(), aggSlot, ignoreResult);
          if (!awaitRes.rv.isIgnored()) {
            // Create the alloca in the block before the scope wrapping
            // cir.await.
            mlir::Value value;
            RValue rv = awaitRes.rv;
            if (rv.isScalar()) {
              value = rv.getValue();
            } else if (rv.isComplex()) {
              value = rv.getComplexValue();
            } else {
              cgf.cgm.errorNYI("emitSuspendExpression: Aggregate value");
              return;
            }

            tmpResumeRValAddr = cgf.emitAlloca(
                "__coawait_resume_rval", value.getType(), loc, CharUnits::One(),
                builder.getBestAllocaInsertPoint(scopeParentBlock));
            // Store the rvalue so we can reload it before the promise call.
            builder.CIRBaseBuilderTy::createStore(loc, value,
                                                  tmpResumeRValAddr);
          }
        }

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
    rval = RValue::getComplex(cir::LoadOp::create(
        cgf.getBuilder(), scopeLoc, rval.getComplexValue().getType(),
        tmpResumeRValAddr));
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
  cir::CoReturnOp::create(builder, loc);

  return mlir::success();
}
