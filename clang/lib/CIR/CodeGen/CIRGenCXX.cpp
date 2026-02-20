//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace clang::CIRGen;

/// Emit code to cause the variable at the given address to be considered as
/// constant from this point onwards.
static void emitDeclInvariant(CIRGenFunction &cgf, const VarDecl *d) {
  mlir::Value addr = cgf.cgm.getAddrOfGlobalVar(d);
  cgf.emitInvariantStart(cgf.getContext().getTypeSizeInChars(d->getType()),
                         addr, cgf.getLoc(d->getSourceRange()));
}

void CIRGenFunction::emitInvariantStart(CharUnits size, mlir::Value addr,
                                        mlir::Location loc) {
  // Do not emit the intrinsic if we're not optimizing.
  if (!cgm.getCodeGenOpts().OptimizationLevel)
    return;

  CIRGenBuilderTy &builder = getBuilder();

  // Create the size constant as i64
  uint64_t width = size.getQuantity();
  mlir::Value sizeValue = builder.getConstInt(loc, builder.getSInt64Ty(),
                                              static_cast<int64_t>(width));

  // Create the intrinsic call. The llvm.invariant.start intrinsic returns a
  // token, but we don't need to capture it. The address space will be
  // automatically handled when the intrinsic is lowered to LLVM IR.
  cir::LLVMIntrinsicCallOp::create(
      builder, loc, builder.getStringAttr("invariant.start"), addr.getType(),
      mlir::ValueRange{sizeValue, addr});
}

static void emitDeclInit(CIRGenFunction &cgf, const VarDecl *varDecl,
                         cir::GlobalOp globalOp) {
  assert((varDecl->hasGlobalStorage() ||
          (varDecl->hasLocalStorage() &&
           cgf.getContext().getLangOpts().OpenCLCPlusPlus)) &&
         "VarDecl must have global or local (in the case of OpenCL) storage!");
  assert(!varDecl->getType()->isReferenceType() &&
         "Should not call emitDeclInit on a reference!");

  CIRGenBuilderTy &builder = cgf.getBuilder();

  // Set up the ctor region.
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block *block = builder.createBlock(&globalOp.getCtorRegion());
  CIRGenFunction::LexicalScope lexScope{cgf, globalOp.getLoc(),
                                        builder.getInsertionBlock()};
  lexScope.setAsGlobalInit();
  builder.setInsertionPointToStart(block);

  Address declAddr(cgf.cgm.getAddrOfGlobalVar(varDecl),
                   cgf.cgm.getASTContext().getDeclAlign(varDecl));

  QualType type = varDecl->getType();
  LValue lv = cgf.makeAddrLValue(declAddr, type);

  const Expr *init = varDecl->getInit();
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case cir::TEK_Scalar:
    assert(!cir::MissingFeatures::objCGC());
    cgf.emitScalarInit(init, cgf.getLoc(varDecl->getLocation()), lv, false);
    break;
  case cir::TEK_Complex:
    cgf.emitComplexExprIntoLValue(init, lv, /*isInit=*/true);
    break;
  case cir::TEK_Aggregate:
    assert(!cir::MissingFeatures::aggValueSlotGC());
    cgf.emitAggExpr(init,
                    AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                            AggValueSlot::IsNotAliased,
                                            AggValueSlot::DoesNotOverlap));
    break;
  }

  // Finish the ctor region.
  builder.setInsertionPointToEnd(block);
  cir::YieldOp::create(builder, globalOp.getLoc());
}

static void emitDeclDestroy(CIRGenFunction &cgf, const VarDecl *vd,
                            cir::GlobalOp addr) {
  // Honor __attribute__((no_destroy)) and bail instead of attempting
  // to emit a reference to a possibly nonexistent destructor, which
  // in turn can cause a crash. This will result in a global constructor
  // that isn't balanced out by a destructor call as intended by the
  // attribute. This also checks for -fno-c++-static-destructors and
  // bails even if the attribute is not present.
  QualType::DestructionKind dtorKind = vd->needsDestruction(cgf.getContext());

  // FIXME:  __attribute__((cleanup)) ?

  switch (dtorKind) {
  case QualType::DK_none:
    return;

  case QualType::DK_cxx_destructor:
    break;

  case QualType::DK_objc_strong_lifetime:
  case QualType::DK_objc_weak_lifetime:
  case QualType::DK_nontrivial_c_struct:
    // We don't care about releasing objects during process teardown.
    assert(!vd->getTLSKind() && "should have rejected this");
    return;
  }

  // If not constant storage we'll emit this regardless of NeedsDtor value.
  CIRGenBuilderTy &builder = cgf.getBuilder();

  // Prepare the dtor region.
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Block *block = builder.createBlock(&addr.getDtorRegion());
  CIRGenFunction::LexicalScope lexScope{cgf, addr.getLoc(),
                                        builder.getInsertionBlock()};
  lexScope.setAsGlobalInit();
  builder.setInsertionPointToStart(block);

  CIRGenModule &cgm = cgf.cgm;
  QualType type = vd->getType();

  // Special-case non-array C++ destructors, if they have the right signature.
  // Under some ABIs, destructors return this instead of void, and cannot be
  // passed directly to __cxa_atexit if the target does not allow this
  // mismatch.
  const CXXRecordDecl *record = type->getAsCXXRecordDecl();
  bool canRegisterDestructor =
      record && (!cgm.getCXXABI().hasThisReturn(
                     GlobalDecl(record->getDestructor(), Dtor_Complete)) ||
                 cgm.getCXXABI().canCallMismatchedFunctionType());

  // If __cxa_atexit is disabled via a flag, a different helper function is
  // generated elsewhere which uses atexit instead, and it takes the destructor
  // directly.
  cir::FuncOp fnOp;
  if (record && (canRegisterDestructor || cgm.getCodeGenOpts().CXAAtExit)) {
    if (vd->getTLSKind())
      cgm.errorNYI(vd->getSourceRange(), "TLS destructor");
    assert(!record->hasTrivialDestructor());
    assert(!cir::MissingFeatures::openCL());
    CXXDestructorDecl *dtor = record->getDestructor();
    // In LLVM OG codegen this is done in registerGlobalDtor, but CIRGen
    // relies on LoweringPrepare for further decoupling, so build the
    // call right here.
    auto gd = GlobalDecl(dtor, Dtor_Complete);
    fnOp = cgm.getAddrAndTypeOfCXXStructor(gd).second;
    builder.createCallOp(cgf.getLoc(vd->getSourceRange()),
                         mlir::FlatSymbolRefAttr::get(fnOp.getSymNameAttr()),
                         mlir::ValueRange{cgm.getAddrOfGlobalVar(vd)});
    assert(fnOp && "expected cir.func");
    // TODO(cir): This doesn't do anything but check for unhandled conditions.
    // What it is meant to do should really be happening in LoweringPrepare.
    cgm.getCXXABI().registerGlobalDtor(vd, fnOp, nullptr);
  } else {
    // Otherwise, a custom destroyed is needed. Classic codegen creates a helper
    // function here and emits the destroy into the helper function, which is
    // called from __cxa_atexit.
    // In CIR, we just emit the destroy into the dtor region. It will be moved
    // into a separate function during the LoweringPrepare pass.
    // FIXME(cir): We should create a new operation here to explicitly get the
    // address of the global into whose dtor region we are emiiting the destroy.
    // The same applies to code above where it is calling getAddrOfGlobalVar.
    mlir::Value globalVal = builder.createGetGlobal(addr);
    CharUnits alignment = cgf.getContext().getDeclAlign(vd);
    Address globalAddr{globalVal, cgf.convertTypeForMem(type), alignment};
    cgf.emitDestroy(globalAddr, type, cgf.getDestroyer(dtorKind));
  }

  builder.setInsertionPointToEnd(block);
  if (block->empty()) {
    block->erase();
    // Don't confuse lexical cleanup.
    builder.clearInsertionPoint();
  } else {
    cir::YieldOp::create(builder, addr.getLoc());
  }
}

cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl gd) {
  const CIRGenFunctionInfo &fnInfo =
      getTypes().arrangeCXXStructorDeclaration(gd);
  cir::FuncType funcType = getTypes().getFunctionType(fnInfo);
  cir::FuncOp fn = getAddrOfCXXStructor(gd, &fnInfo, /*FnType=*/nullptr,
                                        /*DontDefer=*/true, ForDefinition);
  setFunctionLinkage(gd, fn);
  CIRGenFunction cgf{*this, builder};
  curCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.generateCode(gd, fn, funcType);
  }
  curCGF = nullptr;

  setNonAliasAttributes(gd, fn);
  setCIRFunctionAttributesForDefinition(mlir::cast<FunctionDecl>(gd.getDecl()),
                                        fn);
  return fn;
}

// Global variables requiring non-trivial initialization are handled
// differently in CIR than in classic codegen. Classic codegen emits
// a global init function (__cxx_global_var_init) and inserts
// initialization for each global there. In CIR, we attach a ctor
// region to the global variable and insert the initialization code
// into the ctor region. This will be moved into the
// __cxx_global_var_init function during the LoweringPrepare pass.
void CIRGenModule::emitCXXGlobalVarDeclInit(const VarDecl *varDecl,
                                            cir::GlobalOp addr,
                                            bool performInit) {
  QualType ty = varDecl->getType();

  // TODO: handle address space
  // The address space of a static local variable (addr) may be different
  // from the address space of the "this" argument of the constructor. In that
  // case, we need an addrspacecast before calling the constructor.
  //
  // struct StructWithCtor {
  //   __device__ StructWithCtor() {...}
  // };
  // __device__ void foo() {
  //   __shared__ StructWithCtor s;
  //   ...
  // }
  //
  // For example, in the above CUDA code, the static local variable s has a
  // "shared" address space qualifier, but the constructor of StructWithCtor
  // expects "this" in the "generic" address space.
  assert(!cir::MissingFeatures::addressSpace());

  // Create a CIRGenFunction to emit the initializer. While this isn't a true
  // function, the handling works the same way.
  CIRGenFunction cgf{*this, builder, true};
  llvm::SaveAndRestore<CIRGenFunction *> savedCGF(curCGF, &cgf);
  curCGF->curFn = addr;

  CIRGenFunction::SourceLocRAIIObject fnLoc{cgf,
                                            getLoc(varDecl->getLocation())};

  addr.setAstAttr(cir::ASTVarDeclAttr::get(&getMLIRContext(), varDecl));

  if (!ty->isReferenceType()) {
    assert(!cir::MissingFeatures::openMP());

    bool needsDtor = varDecl->needsDestruction(getASTContext()) ==
                     QualType::DK_cxx_destructor;
    bool isConstantStorage =
        varDecl->getType().isConstantStorage(getASTContext(), true, !needsDtor);
    // PerformInit, constant store invariant / destroy handled below.
    if (performInit) {
      emitDeclInit(cgf, varDecl, addr);
      // For constant storage, emit invariant.start in the ctor region after
      // initialization but before the yield.
      if (isConstantStorage) {
        CIRGenBuilderTy &builder = cgf.getBuilder();
        mlir::OpBuilder::InsertionGuard guard(builder);
        // Set insertion point to end of ctor region (before yield)
        if (!addr.getCtorRegion().empty()) {
          mlir::Block *block = &addr.getCtorRegion().back();
          // Find the yield op and insert before it
          mlir::Operation *yieldOp = block->getTerminator();
          if (yieldOp) {
            builder.setInsertionPoint(yieldOp);
            emitDeclInvariant(cgf, varDecl);
          }
        }
      }
    } else if (isConstantStorage) {
      emitDeclInvariant(cgf, varDecl);
    }

    if (!isConstantStorage)
      emitDeclDestroy(cgf, varDecl, addr);
    return;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  auto *block = builder.createBlock(&addr.getCtorRegion());
  CIRGenFunction::LexicalScope scope{*curCGF, addr.getLoc(),
                                     builder.getInsertionBlock()};
  scope.setAsGlobalInit();
  builder.setInsertionPointToStart(block);
  mlir::Value getGlobal = builder.createGetGlobal(addr);

  Address declAddr(getGlobal, getASTContext().getDeclAlign(varDecl));
  assert(performInit && "cannot have a constant initializer which needs "
                        "destruction for reference");
  RValue rv = cgf.emitReferenceBindingToExpr(varDecl->getInit());
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Operation *rvalDefOp = rv.getValue().getDefiningOp();
    if (rvalDefOp && rvalDefOp->getBlock()) {
      mlir::Block *rvalSrcBlock = rvalDefOp->getBlock();

      if (!rvalSrcBlock->empty() && isa<cir::YieldOp>(rvalSrcBlock->back())) {
        mlir::Operation &front = rvalSrcBlock->front();
        getGlobal.getDefiningOp()->moveBefore(&front);
        builder.setInsertionPoint(cast<cir::YieldOp>(rvalSrcBlock->back()));
      }
    }
    cgf.emitStoreOfScalar(rv.getValue(), declAddr, /*isVolatile=*/false,
                          ty, LValueBaseInfo{});
  }

  builder.setInsertionPointToEnd(block);
  cir::YieldOp::create(builder, addr->getLoc());
}
