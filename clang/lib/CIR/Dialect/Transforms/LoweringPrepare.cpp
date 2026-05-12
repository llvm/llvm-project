//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetCXXABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOWERINGPREPARE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

static SmallString<128> getTransformedFileName(mlir::ModuleOp mlirModule) {
  SmallString<128> fileName;

  if (mlirModule.getSymName())
    fileName = llvm::sys::path::filename(mlirModule.getSymName()->str());

  if (fileName.empty())
    fileName = "<null>";

  for (size_t i = 0; i < fileName.size(); ++i) {
    // Replace everything that's not [a-zA-Z0-9._] with a _. This set happens
    // to be the set of C preprocessing numbers.
    if (!clang::isPreprocessingNumberBody(fileName[i]))
      fileName[i] = '_';
  }

  return fileName;
}

namespace {
struct LoweringPreparePass
    : public impl::LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;

  // `mlir::SymbolTableCollection` is move-only (it owns lazily-created
  // `unique_ptr<SymbolTable>` entries), which makes the implicit copy
  // constructor ill-formed.  MLIR's `clonePass()` requires copy
  // construction, so define one explicitly.  Per-run state members
  // (dynamic initializers, guard maps, symbol-table cache, etc.) all
  // start fresh in the cloned pass, which matches MLIR convention for
  // pass clones and is more correct than the previous default-generated
  // behavior that silently copied them.
  LoweringPreparePass(const LoweringPreparePass &other)
      : impl::LoweringPrepareBase<LoweringPreparePass>(other) {}

  void runOnOperation() override;

  void runOnOp(mlir::Operation *op);
  void lowerCastOp(cir::CastOp op);
  void lowerComplexDivOp(cir::ComplexDivOp op);
  void lowerComplexMulOp(cir::ComplexMulOp op);
  void lowerUnaryOp(cir::UnaryOpInterface op);
  void lowerGetGlobalOp(cir::GetGlobalOp op);
  void lowerGlobalOp(cir::GlobalOp op);
  void lowerThreeWayCmpOp(cir::CmpThreeWayOp op);
  void lowerArrayDtor(cir::ArrayDtor op);
  void lowerArrayCtor(cir::ArrayCtor op);
  void lowerTrivialCopyCall(cir::CallOp op);
  void lowerStoreOfConstAggregate(cir::StoreOp op);
  void lowerLocalInitOp(cir::LocalInitOp op);

  /// Return the FuncOp called by `callOp`.  Uses the cached `symbolTables`
  /// member to avoid the O(M) module-wide scan that the static
  /// `mlir::SymbolTable::lookupNearestSymbolFrom` would do per call.
  cir::FuncOp getCalledFunction(cir::CallOp callOp);

  /// Return a private constant cir::GlobalOp with the given type and initial
  /// value, suitable for backing a memcpy-initialized local aggregate.
  ///
  /// If a global with `baseName` (or one of its `.<n>` versioned siblings)
  /// already has a matching type and initial value, that global is reused.
  /// Otherwise a new global is created with the next available `.<n>` suffix
  /// (matching CIRGenBuilder::createVersionedGlobal and OGCG behavior).
  cir::GlobalOp getOrCreateConstAggregateGlobal(CIRBaseBuilderTy &builder,
                                                mlir::Location loc,
                                                llvm::StringRef baseName,
                                                mlir::Type ty,
                                                mlir::TypedAttr constant);

  /// Build the function that initializes the specified global
  cir::FuncOp buildCXXGlobalVarDeclInitFunc(cir::GlobalOp op);

  /// When looking at the 'global' op, create the wrapper function.
  void defineGlobalThreadLocalWrapper(cir::GlobalOp op, cir::FuncOp initAlias,
                                      bool isVarDefinition);
  /// Get the declaration for the 'wrapper' function for a global-TLS variable.
  cir::FuncOp getOrCreateThreadLocalWrapper(CIRBaseBuilderTy &builder,
                                            cir::GlobalOp op);

  /// Handle the dtor region by registering destructor with __cxa_atexit
  cir::FuncOp getOrCreateDtorFunc(CIRBaseBuilderTy &builder, cir::GlobalOp op,
                                  mlir::Region &dtorRegion,
                                  cir::CallOp &dtorCall);

  /// Build a module init function that calls all the dynamic initializers.
  void buildCXXGlobalInitFunc();

  /// Materialize global ctor/dtor list
  void buildGlobalCtorDtorList();

  cir::FuncOp buildRuntimeFunction(
      mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
      cir::FuncType type,
      cir::GlobalLinkageKind linkage = cir::GlobalLinkageKind::ExternalLinkage);

  cir::GlobalOp getOrCreateRuntimeVariable(
      mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
      mlir::Type type,
      cir::GlobalLinkageKind linkage = cir::GlobalLinkageKind::ExternalLinkage,
      cir::VisibilityKind visibility = cir::VisibilityKind::Default);

  /// ------------
  /// CUDA registration related
  /// ------------

  llvm::StringMap<FuncOp> cudaKernelMap;

  /// Build the CUDA module constructor that registers the fat binary
  /// with the CUDA runtime.
  void buildCUDAModuleCtor();
  std::optional<FuncOp> buildCUDAModuleDtor();
  std::optional<FuncOp> buildHIPModuleDtor();
  std::optional<FuncOp> buildCUDARegisterGlobals();
  void buildCUDARegisterGlobalFunctions(cir::CIRBaseBuilderTy &builder,
                                        FuncOp regGlobalFunc);

  /// Handle static local variable initialization with guard variables.
  void handleStaticLocal(cir::GlobalOp globalOp, cir::LocalInitOp localInitOp);

  /// Get or create __cxa_guard_acquire function.
  cir::FuncOp getGuardAcquireFn(cir::PointerType guardPtrTy);

  /// Get or create __cxa_guard_release function.
  cir::FuncOp getGuardReleaseFn(cir::PointerType guardPtrTy);

  /// Create a guard global variable for a static local.
  cir::GlobalOp createGuardGlobalOp(CIRBaseBuilderTy &builder,
                                    mlir::Location loc, llvm::StringRef name,
                                    cir::IntType guardTy,
                                    cir::GlobalLinkageKind linkage);

  /// Get the guard variable for a static local declaration.
  cir::GlobalOp getStaticLocalDeclGuardAddress(llvm::StringRef globalSymName) {
    auto it = staticLocalDeclGuardMap.find(globalSymName);
    if (it != staticLocalDeclGuardMap.end())
      return it->second;
    return nullptr;
  }

  /// Set the guard variable for a static local declaration.
  void setStaticLocalDeclGuardAddress(llvm::StringRef globalSymName,
                                      cir::GlobalOp guard) {
    staticLocalDeclGuardMap[globalSymName] = guard;
  }

  /// Get or create the guard variable for a static local declaration.
  cir::GlobalOp getOrCreateStaticLocalDeclGuardAddress(
      CIRBaseBuilderTy &builder, cir::GlobalOp globalOp,
      cir::ASTVarDeclInterface varDecl, cir::IntType guardTy,
      clang::CharUnits guardAlignment) {
    llvm::StringRef globalSymName = globalOp.getSymName();
    cir::GlobalOp guard = getStaticLocalDeclGuardAddress(globalSymName);
    if (!guard) {
      // Get the guard name from the static_local attribute.
      llvm::StringRef guardName =
          globalOp.getStaticLocalGuard()->getName().getValue();

      // Create the guard variable with a zero-initializer.
      guard = createGuardGlobalOp(builder, globalOp->getLoc(), guardName,
                                  guardTy, globalOp.getLinkage());
      guard.setInitialValueAttr(cir::IntAttr::get(guardTy, 0));
      guard.setDSOLocal(globalOp.getDsoLocal());
      guard.setAlignment(guardAlignment.getAsAlign().value());
      guard.setTlsModel(globalOp.getTlsModel());

      // The ABI says: "It is suggested that it be emitted in the same COMDAT
      // group as the associated data object." In practice, this doesn't work
      // for non-ELF and non-Wasm object formats, so only do it for ELF and
      // Wasm.
      bool hasComdat = globalOp.getComdat();
      const llvm::Triple &triple = astCtx->getTargetInfo().getTriple();
      if (!varDecl.isLocalVarDecl() && hasComdat &&
          (triple.isOSBinFormatELF() || triple.isOSBinFormatWasm())) {
        globalOp->emitError("NYI: guard COMDAT for non-local variables");
        return {};
      } else if (hasComdat && globalOp.isWeakForLinker()) {
        guard.setComdat(true);
      }

      setStaticLocalDeclGuardAddress(globalSymName, guard);
    }
    return guard;
  }

  ///
  /// AST related
  /// -----------

  clang::ASTContext *astCtx;

  /// Tracks current module.
  mlir::ModuleOp mlirModule;

  /// Cached symbol tables used to avoid repeated O(M) module-wide scans
  /// during per-call/per-global symbol lookups.  Lazily populated on first
  /// use.  Pass methods access this directly rather than threading it
  /// through helper signatures (see PR feedback on #195919).
  ///
  /// Invariant: every site that mutates the module's symbol table either
  /// (a) keeps `symbolTables` in sync via
  /// `symbolTables.getSymbolTable(mlirModule).insert(...)` (as
  /// `getOrCreateConstAggregateGlobal` does), or (b) creates a symbol
  /// that is never resolved through the cache later.  Today
  /// `buildRuntimeFunction` and `getOrCreateRuntimeVariable` fall in the
  /// (b) bucket: their callers either use a separate map
  /// (`cudaKernelMap`, `staticLocalDeclGuardMap`, `dynamicInitializers`)
  /// or the static `mlir::SymbolTable::lookupNearestSymbolFrom`, never
  /// the cached path.  If a future change adds a cached lookup of a
  /// freshly created symbol, the corresponding create site MUST move
  /// to bucket (a) (insert into the cache or call
  /// `invalidateSymbolTable`).
  mlir::SymbolTableCollection symbolTables;

  /// Tracks existing dynamic initializers.
  llvm::StringMap<uint32_t> dynamicInitializerNames;
  llvm::SmallVector<cir::FuncOp> dynamicInitializers;
  llvm::StringMap<cir::FuncOp> threadLocalWrappers;

  /// Tracks guard variables for static locals (keyed by global symbol name).
  llvm::StringMap<cir::GlobalOp> staticLocalDeclGuardMap;

  llvm::StringMap<llvm::SmallVector<cir::GlobalOp, 1>> constAggregateGlobals;

  /// List of ctors and their priorities to be called before main()
  llvm::SmallVector<std::pair<std::string, uint32_t>, 4> globalCtorList;
  /// List of dtors and their priorities to be called when unloading module.
  llvm::SmallVector<std::pair<std::string, uint32_t>, 4> globalDtorList;

  /// Returns true if the target uses ARM-style guard variables for static
  /// local initialization (32-bit guard, check bit 0 only).
  bool useARMGuardVarABI() const {
    switch (astCtx->getCXXABIKind()) {
    case clang::TargetCXXABI::GenericARM:
    case clang::TargetCXXABI::iOS:
    case clang::TargetCXXABI::WatchOS:
    case clang::TargetCXXABI::GenericAArch64:
    case clang::TargetCXXABI::WebAssembly:
      return true;
    default:
      return false;
    }
  }

  void emitGlobalGuardedDtorRegion(CIRBaseBuilderTy &builder,
                                   cir::GlobalOp global,
                                   mlir::Region &dtorRegion, bool tls,
                                   mlir::Block &entryBB) {
    // Create a variable that binds the atexit to this shared object.
    builder.setInsertionPointToStart(&mlirModule.getBodyRegion().front());
    cir::GlobalOp handle = getOrCreateRuntimeVariable(
        builder, "__dso_handle", global.getLoc(), builder.getI8Type(),
        cir::GlobalLinkageKind::ExternalLinkage, cir::VisibilityKind::Hidden);

    // If this is a simple call to a destructor, get the called function.
    // Otherwise, create a helper function for the entire dtor region,
    // replacing the current dtor region body with a call to the helper
    // function.
    cir::CallOp dtorCall;
    cir::FuncOp dtorFunc =
        getOrCreateDtorFunc(builder, global, dtorRegion, dtorCall);

    // Create a runtime helper function:
    //    extern "C" int __cxa_atexit(void (*f)(void *), void *p, void *d);
    cir::PointerType voidPtrTy = builder.getVoidPtrTy();
    cir::PointerType voidFnPtrTy = builder.getVoidFnPtrTy({voidPtrTy});
    cir::PointerType handlePtrTy = builder.getPointerTo(handle.getSymType());
    auto fnAtExitType =
        builder.getVoidFnTy({voidFnPtrTy, voidPtrTy, handlePtrTy});

    llvm::StringLiteral nameAtExit = "__cxa_atexit";
    if (tls)
      nameAtExit = astCtx->getTargetInfo().getTriple().isOSDarwin()
                       ? llvm::StringLiteral("_tlv_atexit")
                       : llvm::StringLiteral("__cxa_thread_atexit");

    cir::FuncOp fnAtExit = buildRuntimeFunction(builder, nameAtExit,
                                                global.getLoc(), fnAtExitType);

    // Replace the dtor (or helper) call with a call to
    //   __cxa_atexit(&dtor, &var, &__dso_handle)
    builder.setInsertionPointAfter(dtorCall);
    mlir::Value args[3];
    auto dtorPtrTy = cir::PointerType::get(dtorFunc.getFunctionType());
    args[0] = cir::GetGlobalOp::create(builder, dtorCall.getLoc(), dtorPtrTy,
                                       dtorFunc.getSymName());
    args[0] = cir::CastOp::create(builder, dtorCall.getLoc(), voidFnPtrTy,
                                  cir::CastKind::bitcast, args[0]);
    args[1] =
        cir::CastOp::create(builder, dtorCall.getLoc(), voidPtrTy,
                            cir::CastKind::bitcast, dtorCall.getArgOperand(0));
    args[2] = cir::GetGlobalOp::create(builder, handle.getLoc(), handlePtrTy,
                                       handle.getSymName());
    builder.createCallOp(dtorCall.getLoc(), fnAtExit, args);
    dtorCall->erase();
    mlir::Block &dtorBlock = dtorRegion.front();
    entryBB.getOperations().splice(entryBB.end(), dtorBlock.getOperations(),
                                   dtorBlock.begin(),
                                   std::prev(dtorBlock.end()));
  }

  /// Emit the guarded initialization for a static local variable.
  /// This handles the if/else structure after the guard byte check,
  /// following OG's ItaniumCXXABI::EmitGuardedInit skeleton.
  void emitCXXGuardedInitIf(CIRBaseBuilderTy &builder, cir::GlobalOp globalOp,
                            mlir::Region &ctorRegion, mlir::Region &dtorRegion,
                            cir::ASTVarDeclInterface varDecl,
                            mlir::Value guardPtr, cir::PointerType guardPtrTy,
                            bool threadsafe) {
    auto loc = globalOp->getLoc();

    // The semantics of dynamic initialization of variables with static or
    // thread storage duration depends on whether they are declared at
    // block-scope. The initialization of such variables at block-scope can be
    // aborted with an exception and later retried (per C++20 [stmt.dcl]p4),
    // and recursive entry to their initialization has undefined behavior (also
    // per C++20 [stmt.dcl]p4). For such variables declared at non-block scope,
    // exceptions lead to termination (per C++20 [except.terminate]p1), and
    // recursive references to the variables are governed only by the lifetime
    // rules (per C++20 [class.cdtor]p2), which means such references are
    // perfectly fine as long as they avoid touching memory. As a result,
    // block-scope variables must not be marked as initialized until after
    // initialization completes (unless the mark is reverted following an
    // exception), but non-block-scope variables must be marked prior to
    // initialization so that recursive accesses during initialization do not
    // restart initialization.

    auto emitBody = [&]() {
      // Emit the initializer and add a global destructor if appropriate.
      mlir::Block *insertBlock = builder.getInsertionBlock();
      if (!ctorRegion.empty()) {
        assert(ctorRegion.hasOneBlock() && "Enforced by MaxSizedRegion<1>");

        mlir::Block &block = ctorRegion.front();
        insertBlock->getOperations().splice(
            insertBlock->end(), block.getOperations(), block.begin(),
            std::prev(block.end()));
      }

      if (!dtorRegion.empty()) {
        assert(dtorRegion.hasOneBlock() && "Enforced by MaxSizedRegion<1>");

        emitGlobalGuardedDtorRegion(builder, globalOp, dtorRegion, !threadsafe,
                                    *insertBlock);
      }
      builder.setInsertionPointToEnd(insertBlock);
      ctorRegion.getBlocks().clear();
    };

    // Variables used when coping with thread-safe statics and exceptions.
    if (threadsafe) {
      // Call __cxa_guard_acquire.
      cir::CallOp acquireCall = builder.createCallOp(
          loc, getGuardAcquireFn(guardPtrTy), mlir::ValueRange{guardPtr});
      mlir::Value acquireResult = acquireCall.getResult();

      auto acquireZero = builder.getConstantInt(
          loc, mlir::cast<cir::IntType>(acquireResult.getType()), 0);
      auto shouldInit = builder.createCompare(loc, cir::CmpOpKind::ne,
                                              acquireResult, acquireZero);

      // Create the IfOp for the shouldInit check.
      // Pass an empty callback to avoid auto-creating a yield terminator.
      auto ifOp =
          cir::IfOp::create(builder, loc, shouldInit, /*withElseRegion=*/false,
                            [](mlir::OpBuilder &, mlir::Location) {});
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

      // Call __cxa_guard_abort along the exceptional edge.
      // OG: CGF.EHStack.pushCleanup<CallGuardAbort>(EHCleanup, guard);
      assert(!cir::MissingFeatures::guardAbortOnException());

      emitBody();

      // Pop the guard-abort cleanup if we pushed one.
      // OG: CGF.PopCleanupBlock();
      assert(!cir::MissingFeatures::guardAbortOnException());

      // Call __cxa_guard_release. This cannot throw.
      builder.createCallOp(loc, getGuardReleaseFn(guardPtrTy),
                           mlir::ValueRange{guardPtr});

      builder.createYield(loc);
    } else if (!varDecl.isLocalVarDecl()) {
      // For non-local variables, store 1 into the first byte of the guard
      // variable before the object initialization begins so that references
      // to the variable during initialization don't restart initialization.
      // OG: Builder.CreateStore(llvm::ConstantInt::get(CGM.Int8Ty, 1), ...);
      // Then: CGF.EmitCXXGlobalVarDeclInit(D, var, shouldPerformInit);
      globalOp->emitError("NYI: non-threadsafe init for non-local variables");
      return;
    } else {
      emitBody();
      // For local variables, store 1 into the first byte of the guard variable
      // after the object initialization completes so that initialization is
      // retried if initialization is interrupted by an exception.
      builder.createStore(
          loc, builder.getConstantInt(loc, guardPtrTy.getPointee(), 1),
          guardPtr);
    }

    builder.createYield(loc); // Outermost IfOp
  }

  void setASTContext(clang::ASTContext *c) { astCtx = c; }
};

} // namespace

cir::GlobalOp LoweringPreparePass::getOrCreateRuntimeVariable(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    mlir::Type type, cir::GlobalLinkageKind linkage,
    cir::VisibilityKind visibility) {
  cir::GlobalOp g = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(
          mlirModule, mlir::StringAttr::get(mlirModule->getContext(), name)));
  if (!g) {
    g = cir::GlobalOp::create(builder, loc, name, type);
    g.setLinkageAttr(
        cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
    g.setGlobalVisibility(visibility);
  }
  return g;
}

cir::FuncOp LoweringPreparePass::buildRuntimeFunction(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    cir::FuncType type, cir::GlobalLinkageKind linkage) {
  cir::FuncOp f = dyn_cast_or_null<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      mlirModule, StringAttr::get(mlirModule->getContext(), name)));
  if (!f) {
    f = cir::FuncOp::create(builder, loc, name, type);
    f.setLinkageAttr(
        cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);

    assert(!cir::MissingFeatures::opFuncExtraAttrs());
  }
  return f;
}

static mlir::Value lowerScalarToComplexCast(mlir::MLIRContext &ctx,
                                            cir::CastOp op) {
  cir::CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  mlir::Value imag = builder.getNullValue(src.getType(), op.getLoc());
  return builder.createComplexCreate(op.getLoc(), src, imag);
}

static mlir::Value lowerComplexToScalarCast(mlir::MLIRContext &ctx,
                                            cir::CastOp op,
                                            cir::CastKind elemToBoolKind) {
  cir::CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  if (!mlir::isa<cir::BoolType>(op.getType()))
    return builder.createComplexReal(op.getLoc(), src);

  // Complex cast to bool: (bool)(a+bi) => (bool)a || (bool)b
  mlir::Value srcReal = builder.createComplexReal(op.getLoc(), src);
  mlir::Value srcImag = builder.createComplexImag(op.getLoc(), src);

  cir::BoolType boolTy = builder.getBoolTy();
  mlir::Value srcRealToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcReal, boolTy);
  mlir::Value srcImagToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcImag, boolTy);
  return builder.createLogicalOr(op.getLoc(), srcRealToBool, srcImagToBool);
}

static mlir::Value lowerComplexToComplexCast(mlir::MLIRContext &ctx,
                                             cir::CastOp op,
                                             cir::CastKind scalarCastKind) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  auto dstComplexElemTy =
      mlir::cast<cir::ComplexType>(op.getType()).getElementType();

  mlir::Value srcReal = builder.createComplexReal(op.getLoc(), src);
  mlir::Value srcImag = builder.createComplexImag(op.getLoc(), src);

  mlir::Value dstReal = builder.createCast(op.getLoc(), scalarCastKind, srcReal,
                                           dstComplexElemTy);
  mlir::Value dstImag = builder.createCast(op.getLoc(), scalarCastKind, srcImag,
                                           dstComplexElemTy);
  return builder.createComplexCreate(op.getLoc(), dstReal, dstImag);
}

void LoweringPreparePass::lowerCastOp(cir::CastOp op) {
  mlir::MLIRContext &ctx = getContext();
  mlir::Value loweredValue = [&]() -> mlir::Value {
    switch (op.getKind()) {
    case cir::CastKind::float_to_complex:
    case cir::CastKind::int_to_complex:
      return lowerScalarToComplexCast(ctx, op);
    case cir::CastKind::float_complex_to_real:
    case cir::CastKind::int_complex_to_real:
      return lowerComplexToScalarCast(ctx, op, op.getKind());
    case cir::CastKind::float_complex_to_bool:
      return lowerComplexToScalarCast(ctx, op, cir::CastKind::float_to_bool);
    case cir::CastKind::int_complex_to_bool:
      return lowerComplexToScalarCast(ctx, op, cir::CastKind::int_to_bool);
    case cir::CastKind::float_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::floating);
    case cir::CastKind::float_complex_to_int_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::float_to_int);
    case cir::CastKind::int_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::integral);
    case cir::CastKind::int_complex_to_float_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::int_to_float);
    default:
      return nullptr;
    }
  }();

  if (loweredValue) {
    op.replaceAllUsesWith(loweredValue);
    op.erase();
  }
}

static mlir::Value buildComplexBinOpLibCall(
    LoweringPreparePass &pass, CIRBaseBuilderTy &builder,
    llvm::StringRef (*libFuncNameGetter)(llvm::APFloat::Semantics),
    mlir::Location loc, cir::ComplexType ty, mlir::Value lhsReal,
    mlir::Value lhsImag, mlir::Value rhsReal, mlir::Value rhsImag) {
  cir::FPTypeInterface elementTy =
      mlir::cast<cir::FPTypeInterface>(ty.getElementType());

  llvm::StringRef libFuncName = libFuncNameGetter(
      llvm::APFloat::SemanticsToEnum(elementTy.getFloatSemantics()));
  llvm::SmallVector<mlir::Type, 4> libFuncInputTypes(4, elementTy);

  cir::FuncType libFuncTy = cir::FuncType::get(libFuncInputTypes, ty);

  // Insert a declaration for the runtime function to be used in Complex
  // multiplication and division when needed
  cir::FuncOp libFunc;
  {
    mlir::OpBuilder::InsertionGuard ipGuard{builder};
    builder.setInsertionPointToStart(pass.mlirModule.getBody());
    libFunc = pass.buildRuntimeFunction(builder, libFuncName, loc, libFuncTy);
  }

  cir::CallOp call =
      builder.createCallOp(loc, libFunc, {lhsReal, lhsImag, rhsReal, rhsImag});
  return call.getResult();
}

static llvm::StringRef
getComplexDivLibCallName(llvm::APFloat::Semantics semantics) {
  switch (semantics) {
  case llvm::APFloat::S_IEEEhalf:
    return "__divhc3";
  case llvm::APFloat::S_IEEEsingle:
    return "__divsc3";
  case llvm::APFloat::S_IEEEdouble:
    return "__divdc3";
  case llvm::APFloat::S_PPCDoubleDouble:
    return "__divtc3";
  case llvm::APFloat::S_x87DoubleExtended:
    return "__divxc3";
  case llvm::APFloat::S_IEEEquad:
    return "__divtc3";
  default:
    llvm_unreachable("unsupported floating point type");
  }
}

static mlir::Value
buildAlgebraicComplexDiv(CIRBaseBuilderTy &builder, mlir::Location loc,
                         mlir::Value lhsReal, mlir::Value lhsImag,
                         mlir::Value rhsReal, mlir::Value rhsImag) {
  // (a+bi) / (c+di) = ((ac+bd)/(cc+dd)) + ((bc-ad)/(cc+dd))i
  mlir::Value &a = lhsReal;
  mlir::Value &b = lhsImag;
  mlir::Value &c = rhsReal;
  mlir::Value &d = rhsImag;

  mlir::Value ac = builder.createMul(loc, a, c);     // a*c
  mlir::Value bd = builder.createMul(loc, b, d);     // b*d
  mlir::Value cc = builder.createMul(loc, c, c);     // c*c
  mlir::Value dd = builder.createMul(loc, d, d);     // d*d
  mlir::Value acbd = builder.createAdd(loc, ac, bd); // ac+bd
  mlir::Value ccdd = builder.createAdd(loc, cc, dd); // cc+dd
  mlir::Value resultReal = builder.createDiv(loc, acbd, ccdd);

  mlir::Value bc = builder.createMul(loc, b, c);     // b*c
  mlir::Value ad = builder.createMul(loc, a, d);     // a*d
  mlir::Value bcad = builder.createSub(loc, bc, ad); // bc-ad
  mlir::Value resultImag = builder.createDiv(loc, bcad, ccdd);
  return builder.createComplexCreate(loc, resultReal, resultImag);
}

static mlir::Value
buildRangeReductionComplexDiv(CIRBaseBuilderTy &builder, mlir::Location loc,
                              mlir::Value lhsReal, mlir::Value lhsImag,
                              mlir::Value rhsReal, mlir::Value rhsImag) {
  // Implements Smith's algorithm for complex division.
  // SMITH, R. L. Algorithm 116: Complex division. Commun. ACM 5, 8 (1962).

  // Let:
  //   - lhs := a+bi
  //   - rhs := c+di
  //   - result := lhs / rhs = e+fi
  //
  // The algorithm pseudocode looks like follows:
  //   if fabs(c) >= fabs(d):
  //     r := d / c
  //     tmp := c + r*d
  //     e = (a + b*r) / tmp
  //     f = (b - a*r) / tmp
  //   else:
  //     r := c / d
  //     tmp := d + r*c
  //     e = (a*r + b) / tmp
  //     f = (b*r - a) / tmp

  mlir::Value &a = lhsReal;
  mlir::Value &b = lhsImag;
  mlir::Value &c = rhsReal;
  mlir::Value &d = rhsImag;

  auto trueBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    mlir::Value r = builder.createDiv(loc, d, c);    // r := d / c
    mlir::Value rd = builder.createMul(loc, r, d);   // r*d
    mlir::Value tmp = builder.createAdd(loc, c, rd); // tmp := c + r*d

    mlir::Value br = builder.createMul(loc, b, r);   // b*r
    mlir::Value abr = builder.createAdd(loc, a, br); // a + b*r
    mlir::Value e = builder.createDiv(loc, abr, tmp);

    mlir::Value ar = builder.createMul(loc, a, r);   // a*r
    mlir::Value bar = builder.createSub(loc, b, ar); // b - a*r
    mlir::Value f = builder.createDiv(loc, bar, tmp);

    mlir::Value result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto falseBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    mlir::Value r = builder.createDiv(loc, c, d);    // r := c / d
    mlir::Value rc = builder.createMul(loc, r, c);   // r*c
    mlir::Value tmp = builder.createAdd(loc, d, rc); // tmp := d + r*c

    mlir::Value ar = builder.createMul(loc, a, r);   // a*r
    mlir::Value arb = builder.createAdd(loc, ar, b); // a*r + b
    mlir::Value e = builder.createDiv(loc, arb, tmp);

    mlir::Value br = builder.createMul(loc, b, r);   // b*r
    mlir::Value bra = builder.createSub(loc, br, a); // b*r - a
    mlir::Value f = builder.createDiv(loc, bra, tmp);

    mlir::Value result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto cFabs = cir::FAbsOp::create(builder, loc, c);
  auto dFabs = cir::FAbsOp::create(builder, loc, d);
  cir::CmpOp cmpResult =
      builder.createCompare(loc, cir::CmpOpKind::ge, cFabs, dFabs);
  auto ternary = cir::TernaryOp::create(builder, loc, cmpResult,
                                        trueBranchBuilder, falseBranchBuilder);

  return ternary.getResult();
}

static mlir::Type higherPrecisionElementTypeForComplexArithmetic(
    mlir::MLIRContext &context, clang::ASTContext &cc,
    CIRBaseBuilderTy &builder, mlir::Type elementType) {

  auto getHigherPrecisionFPType = [&context](mlir::Type type) -> mlir::Type {
    if (mlir::isa<cir::FP16Type>(type))
      return cir::SingleType::get(&context);

    if (mlir::isa<cir::SingleType>(type) || mlir::isa<cir::BF16Type>(type))
      return cir::DoubleType::get(&context);

    if (mlir::isa<cir::DoubleType>(type))
      return cir::LongDoubleType::get(&context, type);

    return type;
  };

  auto getFloatTypeSemantics =
      [&cc](mlir::Type type) -> const llvm::fltSemantics & {
    const clang::TargetInfo &info = cc.getTargetInfo();
    if (mlir::isa<cir::FP16Type>(type))
      return info.getHalfFormat();

    if (mlir::isa<cir::BF16Type>(type))
      return info.getBFloat16Format();

    if (mlir::isa<cir::SingleType>(type))
      return info.getFloatFormat();

    if (mlir::isa<cir::DoubleType>(type))
      return info.getDoubleFormat();

    if (mlir::isa<cir::LongDoubleType>(type)) {
      if (cc.getLangOpts().OpenMP && cc.getLangOpts().OpenMPIsTargetDevice)
        llvm_unreachable("NYI Float type semantics with OpenMP");
      return info.getLongDoubleFormat();
    }

    if (mlir::isa<cir::FP128Type>(type)) {
      if (cc.getLangOpts().OpenMP && cc.getLangOpts().OpenMPIsTargetDevice)
        llvm_unreachable("NYI Float type semantics with OpenMP");
      return info.getFloat128Format();
    }

    llvm_unreachable("Unsupported float type semantics");
  };

  const mlir::Type higherElementType = getHigherPrecisionFPType(elementType);
  const llvm::fltSemantics &elementTypeSemantics =
      getFloatTypeSemantics(elementType);
  const llvm::fltSemantics &higherElementTypeSemantics =
      getFloatTypeSemantics(higherElementType);

  // Check that the promoted type can handle the intermediate values without
  // overflowing. This can be interpreted as:
  // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal) * 2 <=
  //      LargerType.LargestFiniteVal.
  // In terms of exponent it gives this formula:
  // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal
  // doubles the exponent of SmallerType.LargestFiniteVal)
  if (llvm::APFloat::semanticsMaxExponent(elementTypeSemantics) * 2 + 1 <=
      llvm::APFloat::semanticsMaxExponent(higherElementTypeSemantics)) {
    return higherElementType;
  }

  // The intermediate values can't be represented in the promoted type
  // without overflowing.
  return {};
}

static mlir::Value
lowerComplexDiv(LoweringPreparePass &pass, CIRBaseBuilderTy &builder,
                mlir::Location loc, cir::ComplexDivOp op, mlir::Value lhsReal,
                mlir::Value lhsImag, mlir::Value rhsReal, mlir::Value rhsImag,
                mlir::MLIRContext &mlirCx, clang::ASTContext &cc) {
  cir::ComplexType complexTy = op.getType();
  if (mlir::isa<cir::FPTypeInterface>(complexTy.getElementType())) {
    cir::ComplexRangeKind range = op.getRange();
    if (range == cir::ComplexRangeKind::Improved)
      return buildRangeReductionComplexDiv(builder, loc, lhsReal, lhsImag,
                                           rhsReal, rhsImag);

    if (range == cir::ComplexRangeKind::Full)
      return buildComplexBinOpLibCall(pass, builder, &getComplexDivLibCallName,
                                      loc, complexTy, lhsReal, lhsImag, rhsReal,
                                      rhsImag);

    if (range == cir::ComplexRangeKind::Promoted) {
      mlir::Type originalElementType = complexTy.getElementType();
      mlir::Type higherPrecisionElementType =
          higherPrecisionElementTypeForComplexArithmetic(mlirCx, cc, builder,
                                                         originalElementType);

      if (!higherPrecisionElementType)
        return buildRangeReductionComplexDiv(builder, loc, lhsReal, lhsImag,
                                             rhsReal, rhsImag);

      cir::CastKind floatingCastKind = cir::CastKind::floating;
      lhsReal = builder.createCast(floatingCastKind, lhsReal,
                                   higherPrecisionElementType);
      lhsImag = builder.createCast(floatingCastKind, lhsImag,
                                   higherPrecisionElementType);
      rhsReal = builder.createCast(floatingCastKind, rhsReal,
                                   higherPrecisionElementType);
      rhsImag = builder.createCast(floatingCastKind, rhsImag,
                                   higherPrecisionElementType);

      mlir::Value algebraicResult = buildAlgebraicComplexDiv(
          builder, loc, lhsReal, lhsImag, rhsReal, rhsImag);

      mlir::Value resultReal = builder.createComplexReal(loc, algebraicResult);
      mlir::Value resultImag = builder.createComplexImag(loc, algebraicResult);

      mlir::Value finalReal =
          builder.createCast(floatingCastKind, resultReal, originalElementType);
      mlir::Value finalImag =
          builder.createCast(floatingCastKind, resultImag, originalElementType);
      return builder.createComplexCreate(loc, finalReal, finalImag);
    }
  }

  return buildAlgebraicComplexDiv(builder, loc, lhsReal, lhsImag, rhsReal,
                                  rhsImag);
}

void LoweringPreparePass::lowerComplexDivOp(cir::ComplexDivOp op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);
  mlir::Location loc = op.getLoc();
  mlir::TypedValue<cir::ComplexType> lhs = op.getLhs();
  mlir::TypedValue<cir::ComplexType> rhs = op.getRhs();
  mlir::Value lhsReal = builder.createComplexReal(loc, lhs);
  mlir::Value lhsImag = builder.createComplexImag(loc, lhs);
  mlir::Value rhsReal = builder.createComplexReal(loc, rhs);
  mlir::Value rhsImag = builder.createComplexImag(loc, rhs);

  mlir::Value loweredResult =
      lowerComplexDiv(*this, builder, loc, op, lhsReal, lhsImag, rhsReal,
                      rhsImag, getContext(), *astCtx);
  op.replaceAllUsesWith(loweredResult);
  op.erase();
}

static llvm::StringRef
getComplexMulLibCallName(llvm::APFloat::Semantics semantics) {
  switch (semantics) {
  case llvm::APFloat::S_IEEEhalf:
    return "__mulhc3";
  case llvm::APFloat::S_IEEEsingle:
    return "__mulsc3";
  case llvm::APFloat::S_IEEEdouble:
    return "__muldc3";
  case llvm::APFloat::S_PPCDoubleDouble:
    return "__multc3";
  case llvm::APFloat::S_x87DoubleExtended:
    return "__mulxc3";
  case llvm::APFloat::S_IEEEquad:
    return "__multc3";
  default:
    llvm_unreachable("unsupported floating point type");
  }
}

static mlir::Value lowerComplexMul(LoweringPreparePass &pass,
                                   CIRBaseBuilderTy &builder,
                                   mlir::Location loc, cir::ComplexMulOp op,
                                   mlir::Value lhsReal, mlir::Value lhsImag,
                                   mlir::Value rhsReal, mlir::Value rhsImag) {
  // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
  mlir::Value resultRealLhs = builder.createMul(loc, lhsReal, rhsReal); // ac
  mlir::Value resultRealRhs = builder.createMul(loc, lhsImag, rhsImag); // bd
  mlir::Value resultImagLhs = builder.createMul(loc, lhsReal, rhsImag); // ad
  mlir::Value resultImagRhs = builder.createMul(loc, lhsImag, rhsReal); // bc
  mlir::Value resultReal = builder.createSub(loc, resultRealLhs, resultRealRhs);
  mlir::Value resultImag = builder.createAdd(loc, resultImagLhs, resultImagRhs);
  mlir::Value algebraicResult =
      builder.createComplexCreate(loc, resultReal, resultImag);

  cir::ComplexType complexTy = op.getType();
  cir::ComplexRangeKind rangeKind = op.getRange();
  if (mlir::isa<cir::IntType>(complexTy.getElementType()) ||
      rangeKind == cir::ComplexRangeKind::Basic ||
      rangeKind == cir::ComplexRangeKind::Improved ||
      rangeKind == cir::ComplexRangeKind::Promoted)
    return algebraicResult;

  assert(!cir::MissingFeatures::fastMathFlags());

  // Check whether the real part and the imaginary part of the result are both
  // NaN. If so, emit a library call to compute the multiplication instead.
  // We check a value against NaN by comparing the value against itself.
  mlir::Value resultRealIsNaN = builder.createIsNaN(loc, resultReal);
  mlir::Value resultImagIsNaN = builder.createIsNaN(loc, resultImag);
  mlir::Value resultRealAndImagAreNaN =
      builder.createLogicalAnd(loc, resultRealIsNaN, resultImagIsNaN);

  return cir::TernaryOp::create(
             builder, loc, resultRealAndImagAreNaN,
             [&](mlir::OpBuilder &, mlir::Location) {
               mlir::Value libCallResult = buildComplexBinOpLibCall(
                   pass, builder, &getComplexMulLibCallName, loc, complexTy,
                   lhsReal, lhsImag, rhsReal, rhsImag);
               builder.createYield(loc, libCallResult);
             },
             [&](mlir::OpBuilder &, mlir::Location) {
               builder.createYield(loc, algebraicResult);
             })
      .getResult();
}

void LoweringPreparePass::lowerComplexMulOp(cir::ComplexMulOp op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);
  mlir::Location loc = op.getLoc();
  mlir::TypedValue<cir::ComplexType> lhs = op.getLhs();
  mlir::TypedValue<cir::ComplexType> rhs = op.getRhs();
  mlir::Value lhsReal = builder.createComplexReal(loc, lhs);
  mlir::Value lhsImag = builder.createComplexImag(loc, lhs);
  mlir::Value rhsReal = builder.createComplexReal(loc, rhs);
  mlir::Value rhsImag = builder.createComplexImag(loc, rhs);
  mlir::Value loweredResult = lowerComplexMul(*this, builder, loc, op, lhsReal,
                                              lhsImag, rhsReal, rhsImag);
  op.replaceAllUsesWith(loweredResult);
  op.erase();
}

void LoweringPreparePass::lowerUnaryOp(cir::UnaryOpInterface op) {
  if (!mlir::isa<cir::ComplexType>(op.getResult().getType()))
    return;

  mlir::Location loc = op->getLoc();
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  mlir::Value operand = op.getInput();
  mlir::Value operandReal = builder.createComplexReal(loc, operand);
  mlir::Value operandImag = builder.createComplexImag(loc, operand);

  mlir::Value resultReal = operandReal;
  mlir::Value resultImag = operandImag;

  llvm::TypeSwitch<mlir::Operation *>(op)
      .Case<cir::IncOp>(
          [&](auto) { resultReal = builder.createInc(loc, operandReal); })
      .Case<cir::DecOp>(
          [&](auto) { resultReal = builder.createDec(loc, operandReal); })
      .Case<cir::MinusOp>([&](auto) {
        resultReal = builder.createMinus(loc, operandReal);
        resultImag = builder.createMinus(loc, operandImag);
      })
      .Case<cir::NotOp>(
          [&](auto) { resultImag = builder.createMinus(loc, operandImag); })
      .Default([](auto) { llvm_unreachable("unhandled unary complex op"); });

  mlir::Value result = builder.createComplexCreate(loc, resultReal, resultImag);
  op->replaceAllUsesWith(mlir::ValueRange{result});
  op->erase();
}

cir::FuncOp LoweringPreparePass::getOrCreateDtorFunc(CIRBaseBuilderTy &builder,
                                                     cir::GlobalOp op,
                                                     mlir::Region &dtorRegion,
                                                     cir::CallOp &dtorCall) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  assert(!cir::MissingFeatures::astVarDeclInterface());
  assert(!cir::MissingFeatures::opGlobalThreadLocal());

  cir::VoidType voidTy = builder.getVoidTy();
  auto voidPtrTy = cir::PointerType::get(voidTy);

  // Look for operations in dtorBlock
  mlir::Block &dtorBlock = dtorRegion.front();

  // The first operation should be a get_global to retrieve the address
  // of the global variable we're destroying.
  auto opIt = dtorBlock.getOperations().begin();
  cir::GetGlobalOp ggop = mlir::cast<cir::GetGlobalOp>(*opIt);

  // The simple case is just a call to a destructor, like this:
  //
  //   %0 = cir.get_global %globalS : !cir.ptr<!rec_S>
  //   cir.call %_ZN1SD1Ev(%0) : (!cir.ptr<!rec_S>) -> ()
  //   (implicit cir.yield)
  //
  // That is, if the second operation is a call that takes the get_global result
  // as its only operand, and the only other operation is a yield, then we can
  // just return the called function.
  if (dtorBlock.getOperations().size() == 3) {
    auto callOp = mlir::dyn_cast<cir::CallOp>(&*(++opIt));
    auto yieldOp = mlir::dyn_cast<cir::YieldOp>(&*(++opIt));
    if (yieldOp && callOp && callOp.getNumOperands() == 1 &&
        callOp.getArgOperand(0) == ggop) {
      dtorCall = callOp;
      return getCalledFunction(callOp);
    }
  }

  // Otherwise, we need to create a helper function to replace the dtor region.
  // This name is kind of arbitrary, but it matches the name that classic
  // codegen uses, based on the expected case that gets us here.
  builder.setInsertionPointAfter(op);
  SmallString<256> fnName("__cxx_global_array_dtor");
  uint32_t cnt = dynamicInitializerNames[fnName]++;
  if (cnt)
    fnName += "." + std::to_string(cnt);

  // Create the helper function.
  auto fnType = cir::FuncType::get({voidPtrTy}, voidTy);
  cir::FuncOp dtorFunc =
      buildRuntimeFunction(builder, fnName, op.getLoc(), fnType,
                           cir::GlobalLinkageKind::InternalLinkage);

  SmallVector<mlir::NamedAttribute> paramAttrs;
  paramAttrs.push_back(
      builder.getNamedAttr("llvm.noundef", builder.getUnitAttr()));
  SmallVector<mlir::Attribute> argAttrDicts;
  argAttrDicts.push_back(
      mlir::DictionaryAttr::get(builder.getContext(), paramAttrs));
  dtorFunc.setArgAttrsAttr(
      mlir::ArrayAttr::get(builder.getContext(), argAttrDicts));

  mlir::Block *entryBB = dtorFunc.addEntryBlock();

  // Move everything from the dtor region into the helper function.
  entryBB->getOperations().splice(entryBB->begin(), dtorBlock.getOperations(),
                                  dtorBlock.begin(), dtorBlock.end());

  // Before erasing this, clone it back into the dtor region
  cir::GetGlobalOp dtorGGop =
      mlir::cast<cir::GetGlobalOp>(entryBB->getOperations().front());
  builder.setInsertionPointToStart(&dtorBlock);
  builder.clone(*dtorGGop.getOperation());

  // Replace all uses of the help function's get_global with the function
  // argument.
  mlir::Value dtorArg = entryBB->getArgument(0);
  dtorGGop.replaceAllUsesWith(dtorArg);
  dtorGGop.erase();

  // Replace the yield in the final block with a return
  mlir::Block &finalBlock = dtorFunc.getBody().back();
  auto yieldOp = cast<cir::YieldOp>(finalBlock.getTerminator());
  builder.setInsertionPoint(yieldOp);
  cir::ReturnOp::create(builder, yieldOp->getLoc());
  yieldOp->erase();

  // Create a call to the helper function, passing the original get_global op
  // as the argument.
  cir::GetGlobalOp origGGop =
      mlir::cast<cir::GetGlobalOp>(dtorBlock.getOperations().front());
  builder.setInsertionPointAfter(origGGop);
  mlir::Value ggopResult = origGGop.getResult();
  dtorCall = builder.createCallOp(op.getLoc(), dtorFunc, ggopResult);

  // Add a yield after the call.
  auto finalYield = cir::YieldOp::create(builder, op.getLoc());

  // Erase everything after the yield.
  dtorBlock.getOperations().erase(std::next(mlir::Block::iterator(finalYield)),
                                  dtorBlock.end());
  dtorRegion.getBlocks().erase(std::next(dtorRegion.begin()), dtorRegion.end());

  return dtorFunc;
}

cir::FuncOp
LoweringPreparePass::buildCXXGlobalVarDeclInitFunc(cir::GlobalOp op) {
  // TODO(cir): Store this in the GlobalOp.
  // This should come from the MangleContext, but for now I'm hardcoding it.
  SmallString<256> fnName("__cxx_global_var_init");
  // Get a unique name
  uint32_t cnt = dynamicInitializerNames[fnName]++;
  if (cnt)
    fnName += "." + std::to_string(cnt);

  // Create a variable initialization function.
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);
  cir::VoidType voidTy = builder.getVoidTy();
  auto fnType = cir::FuncType::get({}, voidTy);
  FuncOp f = buildRuntimeFunction(builder, fnName, op.getLoc(), fnType,
                                  cir::GlobalLinkageKind::InternalLinkage);

  // Move over the initialization code of the ctor region.
  // The ctor region may have multiple blocks when exception handling
  // scaffolding creates extra blocks (e.g., unreachable/trap blocks).
  // We move all operations from the first block (minus the yield) into
  // the function entry, and discard extra blocks (which contain only
  // unreachable terminators from EH cleanup paths).
  mlir::Block *entryBB = f.addEntryBlock();
  if (!op.getCtorRegion().empty()) {
    mlir::Block &block = op.getCtorRegion().front();
    entryBB->getOperations().splice(entryBB->begin(), block.getOperations(),
                                    block.begin(), std::prev(block.end()));
  }

  // Register the destructor call with __cxa_atexit
  mlir::Region &dtorRegion = op.getDtorRegion();
  if (!dtorRegion.empty()) {
    assert(!cir::MissingFeatures::astVarDeclInterface());
    assert(!cir::MissingFeatures::opGlobalThreadLocal());

    emitGlobalGuardedDtorRegion(builder, op, dtorRegion,
                                op.getTlsModel().has_value(), *entryBB);
  }

  // Replace cir.yield with cir.return
  builder.setInsertionPointToEnd(entryBB);
  mlir::Operation *yieldOp = nullptr;
  if (!op.getCtorRegion().empty()) {
    mlir::Block &block = op.getCtorRegion().front();
    yieldOp = &block.getOperations().back();
  } else {
    assert(!dtorRegion.empty());
    mlir::Block &block = dtorRegion.front();
    yieldOp = &block.getOperations().back();
  }

  assert(isa<cir::YieldOp>(*yieldOp));
  cir::ReturnOp::create(builder, yieldOp->getLoc());
  return f;
}

cir::FuncOp
LoweringPreparePass::getGuardAcquireFn(cir::PointerType guardPtrTy) {
  // int __cxa_guard_acquire(__guard *guard_object);
  CIRBaseBuilderTy builder(getContext());
  mlir::OpBuilder::InsertionGuard ipGuard{builder};
  builder.setInsertionPointToStart(mlirModule.getBody());
  mlir::Location loc = mlirModule.getLoc();
  cir::IntType intTy = cir::IntType::get(&getContext(), 32, /*isSigned=*/true);
  auto fnType = cir::FuncType::get({guardPtrTy}, intTy);
  return buildRuntimeFunction(builder, "__cxa_guard_acquire", loc, fnType);
}

cir::FuncOp
LoweringPreparePass::getGuardReleaseFn(cir::PointerType guardPtrTy) {
  // void __cxa_guard_release(__guard *guard_object);
  CIRBaseBuilderTy builder(getContext());
  mlir::OpBuilder::InsertionGuard ipGuard{builder};
  builder.setInsertionPointToStart(mlirModule.getBody());
  mlir::Location loc = mlirModule.getLoc();
  cir::VoidType voidTy = cir::VoidType::get(&getContext());
  auto fnType = cir::FuncType::get({guardPtrTy}, voidTy);
  return buildRuntimeFunction(builder, "__cxa_guard_release", loc, fnType);
}

cir::GlobalOp LoweringPreparePass::createGuardGlobalOp(
    CIRBaseBuilderTy &builder, mlir::Location loc, llvm::StringRef name,
    cir::IntType guardTy, cir::GlobalLinkageKind linkage) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mlirModule.getBody());
  cir::GlobalOp g = cir::GlobalOp::create(builder, loc, name, guardTy);
  g.setLinkageAttr(
      cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
  mlir::SymbolTable::setSymbolVisibility(
      g, mlir::SymbolTable::Visibility::Private);
  return g;
}

void LoweringPreparePass::handleStaticLocal(cir::GlobalOp globalOp,
                                            cir::LocalInitOp localInitOp) {
  CIRBaseBuilderTy builder(getContext());

  std::optional<cir::ASTVarDeclInterface> astOption = globalOp.getAst();
  assert(astOption.has_value());
  cir::ASTVarDeclInterface varDecl = astOption.value();

  builder.setInsertionPointAfter(localInitOp);
  mlir::Block *localInitBlock = builder.getInsertionBlock();

  // Remove the terminator temporarily - we'll add it back at the end.
  mlir::Operation *ret = localInitBlock->getTerminator();
  ret->remove();
  // Note: These two insert-point-after sets are necessary, as the 'trailing'
  // operation has changed thanks to the terminator removal.
  builder.setInsertionPointAfter(localInitOp);

  // Inline variables that weren't instantiated from variable templates have
  // partially-ordered initialization within their translation unit.
  bool nonTemplateInline =
      varDecl.isInline() &&
      !clang::isTemplateInstantiation(varDecl.getTemplateSpecializationKind());

  // Inline namespace-scope variables require guarded initialization in a
  // __cxx_global_var_init function. This is not yet implemented.
  if (nonTemplateInline) {
    globalOp->emitError(
        "NYI: guarded initialization for inline namespace-scope variables");
    return;
  }

  // We only need to use thread-safe statics for local non-TLS variables and
  // inline variables; other global initialization is always single-threaded
  // or (through lazy dynamic loading in multiple threads) unsequenced.
  bool threadsafe = astCtx->getLangOpts().ThreadsafeStatics &&
                    (varDecl.isLocalVarDecl() || nonTemplateInline) &&
                    !varDecl.getTLSKind();

  // If we have a global variable with internal linkage and thread-safe statics
  // are disabled, we can just let the guard variable be of type i8.
  bool useInt8GuardVariable = !threadsafe && globalOp.hasInternalLinkage();
  cir::CIRDataLayout dataLayout(mlirModule);
  cir::IntType guardTy;
  clang::CharUnits guardAlignment;
  // Guard variables are 64 bits in the generic ABI and size width on ARM
  // (i.e. 32-bit on AArch32, 64-bit on AArch64).
  if (useInt8GuardVariable) {
    guardTy = cir::IntType::get(&getContext(), 8, /*isSigned=*/true);
    guardAlignment = clang::CharUnits::One();
  } else if (useARMGuardVarABI()) {
    // Guard variables are size width on ARM (32-bit AArch32, 64-bit AArch64).
    const unsigned sizeTypeSize =
        astCtx->getTypeSize(astCtx->getSignedSizeType());
    guardTy = cir::IntType::get(&getContext(), sizeTypeSize, /*isSigned=*/true);
    guardAlignment =
        clang::CharUnits::fromQuantity(dataLayout.getABITypeAlign(guardTy));
  } else {
    guardTy = cir::IntType::get(&getContext(), 64, /*isSigned=*/true);
    guardAlignment =
        clang::CharUnits::fromQuantity(dataLayout.getABITypeAlign(guardTy));
  }
  assert(guardTy && guardAlignment.getQuantity() != 0);

  auto guardPtrTy = cir::PointerType::get(guardTy);

  // Create the guard variable if we don't already have it.
  cir::GlobalOp guard = getOrCreateStaticLocalDeclGuardAddress(
      builder, globalOp, varDecl, guardTy, guardAlignment);
  if (!guard) {
    // Error was already emitted, just restore the terminator and return.
    localInitBlock->push_back(ret);
    return;
  }

  mlir::Value guardPtr = builder.createGetGlobal(guard, localInitOp.getTls());

  // Test whether the variable has completed initialization.
  //
  // Itanium C++ ABI 3.3.2:
  //   The following is pseudo-code showing how these functions can be used:
  //     if (obj_guard.first_byte == 0) {
  //       if ( __cxa_guard_acquire (&obj_guard) ) {
  //         try {
  //           ... initialize the object ...;
  //         } catch (...) {
  //            __cxa_guard_abort (&obj_guard);
  //            throw;
  //         }
  //         ... queue object destructor with __cxa_atexit() ...;
  //         __cxa_guard_release (&obj_guard);
  //       }
  //     }
  //
  // If threadsafe statics are enabled, but we don't have inline atomics, just
  // call __cxa_guard_acquire unconditionally. The "inline" check isn't
  // actually inline, and the user might not expect calls to __atomic libcalls.
  unsigned maxInlineWidthInBits =
      astCtx->getTargetInfo().getMaxAtomicInlineWidth();

  if (!threadsafe || maxInlineWidthInBits) {
    // Load the first byte of the guard variable.
    auto bytePtrTy = cir::PointerType::get(builder.getSIntNTy(8));
    mlir::Value bytePtr = builder.createBitcast(guardPtr, bytePtrTy);
    mlir::Value guardLoad = builder.createAlignedLoad(
        localInitOp.getLoc(), bytePtr, guardAlignment.getAsAlign().value());

    // Itanium ABI:
    //   An implementation supporting thread-safety on multiprocessor
    //   systems must also guarantee that references to the initialized
    //   object do not occur before the load of the initialization flag.
    //
    // In LLVM, we do this by marking the load Acquire.
    if (threadsafe) {
      auto loadOp = mlir::cast<cir::LoadOp>(guardLoad.getDefiningOp());
      loadOp.setMemOrder(cir::MemOrder::Acquire);
      loadOp.setSyncScope(cir::SyncScopeKind::System);
    }

    // For ARM, we should only check the first bit, rather than the entire byte:
    //
    // ARM C++ ABI 3.2.3.1:
    //   To support the potential use of initialization guard variables
    //   as semaphores that are the target of ARM SWP and LDREX/STREX
    //   synchronizing instructions we define a static initialization
    //   guard variable to be a 4-byte aligned, 4-byte word with the
    //   following inline access protocol.
    //     #define INITIALIZED 1
    //     if ((obj_guard & INITIALIZED) != INITIALIZED) {
    //       if (__cxa_guard_acquire(&obj_guard))
    //         ...
    //     }
    //
    // and similarly for ARM64:
    //
    // ARM64 C++ ABI 3.2.2:
    //   This ABI instead only specifies the value bit 0 of the static guard
    //   variable; all other bits are platform defined. Bit 0 shall be 0 when
    //   the variable is not initialized and 1 when it is.
    if (useARMGuardVarABI() && !useInt8GuardVariable) {
      auto one = builder.getConstantInt(
          localInitOp.getLoc(), mlir::cast<cir::IntType>(guardLoad.getType()),
          1);
      guardLoad = builder.createAnd(localInitOp.getLoc(), guardLoad, one);
    }

    // Check if the first byte of the guard variable is zero.
    auto zero = builder.getConstantInt(
        localInitOp.getLoc(), mlir::cast<cir::IntType>(guardLoad.getType()), 0);
    auto needsInit = builder.createCompare(localInitOp.getLoc(),
                                           cir::CmpOpKind::eq, guardLoad, zero);

    // Build the guarded initialization inside an if block.
    cir::IfOp::create(
        builder, globalOp.getLoc(), needsInit,
        /*withElseRegion=*/false, [&](mlir::OpBuilder &, mlir::Location) {
          emitCXXGuardedInitIf(builder, globalOp, localInitOp.getCtorRegion(),
                               localInitOp.getDtorRegion(), varDecl, guardPtr,
                               guardPtrTy, threadsafe);
        });
  } else {
    // Threadsafe statics without inline atomics - call __cxa_guard_acquire
    // unconditionally without the initial guard byte check.
    globalOp->emitError("NYI: guarded init without inline atomics support");
    return;
  }

  // Insert the removed terminator back.
  builder.getInsertionBlock()->push_back(ret);
}

void LoweringPreparePass::lowerLocalInitOp(cir::LocalInitOp initOp) {

  // If we don't actually need to initialize anything anymore, we're done here.
  if (initOp.getCtorRegion().empty() && initOp.getDtorRegion().empty()) {
    initOp.erase();
    return;
  }

  cir::GlobalOp globalOp = initOp.getReferencedGlobal(symbolTables);
  assert(globalOp && "No global-op found");

  handleStaticLocal(globalOp, initOp);

  // Remove the init local op, now that we've done everything we need with it.
  initOp.erase();
}
static bool isThreadWrapperReplaceable(cir::TLS_Model tls,
                                       clang::ASTContext &astCtx) {
  return tls == cir::TLS_Model::GeneralDynamic &&
         astCtx.getTargetInfo().getTriple().isOSDarwin();
}

static cir::GlobalLinkageKind
getThreadLocalWrapperLinkage(GlobalOp op, clang::ASTContext &astCtx) {
  if (isLocalLinkage(op.getLinkage()))
    return op.getLinkage();

  if (isThreadWrapperReplaceable(*op.getTlsModel(), astCtx))
    if (!isLinkOnceLinkage(op.getLinkage()) &&
        !isWeakODRLinkage(op.getLinkage()))
      return op.getLinkage();

  // If this isn't a TU in which this variable is defined, the thread wrapper is
  // discardable.
  if (op.isDeclaration())
    return cir::GlobalLinkageKind::LinkOnceODRLinkage;
  return cir::GlobalLinkageKind::WeakODRLinkage;
}

cir::FuncOp
LoweringPreparePass::getOrCreateThreadLocalWrapper(CIRBaseBuilderTy &builder,
                                                   GlobalOp op) {
  mlir::OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(&mlirModule.getBodyRegion().front());

  mlir::StringAttr wrapperName = op.getDynTlsRefs()->getWrapperName();

  auto existingWrapperIter = threadLocalWrappers.find(wrapperName.getValue());
  if (existingWrapperIter != threadLocalWrappers.end())
    return existingWrapperIter->second;

  // type is ptr-to-global-type(void);
  auto funcType = cir::FuncType::get({}, builder.getPointerTo(op.getSymType()));
  cir::FuncOp func =
      cir::FuncOp::create(builder, op.getLoc(), wrapperName, funcType);

  cir::GlobalLinkageKind linkageKind =
      getThreadLocalWrapperLinkage(op, *astCtx);
  func.setLinkageAttr(
      cir::GlobalLinkageKindAttr::get(&getContext(), linkageKind));

  // TODO(cir): This is supposed to refer to the comdat of the global symbol,
  // but that isn't in CIR yet.
  if (astCtx->getTargetInfo().getTriple().supportsCOMDAT() &&
      func.isWeakForLinker())
    func.setComdat(true);

  mlir::SymbolTable::setSymbolVisibility(
      func, mlir::SymbolTable::Visibility::Private);

  if (!isLocalLinkage(linkageKind)) {
    if (!isThreadWrapperReplaceable(*op.getTlsModel(), *astCtx) ||
        isLinkOnceLinkage(linkageKind) || isWeakODRLinkage(linkageKind) ||
        op.getGlobalVisibility() == cir::VisibilityKind::Hidden)
      func.setGlobalVisibility(cir::VisibilityKind::Hidden);
  }
  if (isThreadWrapperReplaceable(*op.getTlsModel(), *astCtx))
    op->emitError("Unhandled thread wrapper attributes for CC and Nounwind");

  threadLocalWrappers.insert({wrapperName.getValue(), func});
  return func;
}

void LoweringPreparePass::defineGlobalThreadLocalWrapper(cir::GlobalOp op,
                                                         cir::FuncOp initAlias,
                                                         bool isVarDefinition) {
  CIRBaseBuilderTy builder(getContext());
  cir::FuncOp wrapper = getOrCreateThreadLocalWrapper(builder, op);
  mlir::Block *entryBB = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(entryBB);
  // If we are a situation where we have/need one, emit a call to the init
  // function.
  if (initAlias) {
    op->emitError("not yet implemented, wrapper with an init alias");
  }
  auto get = builder.createGetGlobal(op, /*tls=*/true);
  cir::ReturnOp::create(builder, op.getLoc(), {get});
}

void LoweringPreparePass::lowerGlobalOp(GlobalOp op) {
  // Static locals are handled separately via guard variables.
  if (op.getStaticLocalGuard())
    return;

  mlir::Region &ctorRegion = op.getCtorRegion();
  mlir::Region &dtorRegion = op.getDtorRegion();
  // TODO(cir): Implement the initialization of this.
  cir::FuncOp initAlias;

  if (!ctorRegion.empty() || !dtorRegion.empty()) {
    // Build a variable initialization function and move the initialzation code
    // in the ctor region over.
    cir::FuncOp f = buildCXXGlobalVarDeclInitFunc(op);

    // Clear the ctor and dtor region
    ctorRegion.getBlocks().clear();
    dtorRegion.getBlocks().clear();

    assert(!cir::MissingFeatures::astVarDeclInterface());
    dynamicInitializers.push_back(f);
  }

  // We need a wrapper for TLS globals that MIGHT have a non-constant
  // initialization. The FE will have generated the DynTlsRefs for any with
  // known dynamic init, or unknown (extern) init.
  if (op.getTlsModel() == TLS_Model::GeneralDynamic && op.getDynTlsRefs())
    defineGlobalThreadLocalWrapper(op, initAlias, !op.isDeclaration());

  assert(!cir::MissingFeatures::opGlobalAnnotations());
}

void LoweringPreparePass::lowerGetGlobalOp(GetGlobalOp op) {
  if (!op.getTls())
    return;
  auto globalOp = mlir::cast<cir::GlobalOp>(
      symbolTables.lookupNearestSymbolFrom(op, op.getNameAttr()));

  // Only global/namespace scope thread local variables need to have their
  // get-global operations rewritten to be calls to a wrapper function.  If
  // we're not in a dynamic TLS (or one without the TLS markers), we can leave
  // this one as a get-global and return early.
  if (globalOp.getTlsModel() != TLS_Model::GeneralDynamic ||
      !globalOp.getDynTlsRefs())
    return;

  // If this is a global TLS, we need to replace the call to 'get_global' with a
  // call to the wrapper function.  Classic codegen figures out some cases where
  // we can omit this, but for now we're going to always put it in, as it is
  // effectively a no-op.

  // The first 'GetGlobalOp' at the beginning of a ctor/dtor region on one of
  // these is for the purpose of creating/destroying.  We want to skip replacing
  // THAT one, but leave all other get-global-ops in place, else
  // self-referential ops won't work right.

  // Note that ctors/dtors are removed during this pass. We get away with these
  // checks because the only time that these situations can actually be true
  // (that is, the ctor/dtor region exist) is if we're in the process of
  // converting the ctor/dtor for this. If we're NOT doing that, the ctor/dtor
  // will have already disappeared.
  mlir::Operation *parentOp = op->getParentOp();
  if (parentOp == globalOp) {
    mlir::Region *ctorRegion = &globalOp.getCtorRegion();
    mlir::Region *dtorRegion = &globalOp.getDtorRegion();

    if (!ctorRegion->empty() && &*ctorRegion->op_begin() == op.getOperation())
      return;
    if (!dtorRegion->empty() && &*dtorRegion->op_begin() == op.getOperation())
      return;
  }

  CIRBaseBuilderTy builder(getContext());
  cir::FuncOp wrapperFunc = getOrCreateThreadLocalWrapper(builder, globalOp);

  builder.setInsertionPoint(op);
  cir::CallOp call = builder.createCallOp(
      wrapperFunc.getLoc(),
      mlir::FlatSymbolRefAttr::get(wrapperFunc.getSymNameAttr()),
      wrapperFunc.getFunctionType().getReturnType(), {});
  op->replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::lowerThreeWayCmpOp(CmpThreeWayOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  mlir::Location loc = op->getLoc();
  cir::CmpThreeWayInfoAttr cmpInfo = op.getInfo();

  mlir::Value ltRes =
      builder.getConstantInt(loc, op.getType(), cmpInfo.getLt());
  mlir::Value eqRes =
      builder.getConstantInt(loc, op.getType(), cmpInfo.getEq());
  mlir::Value gtRes =
      builder.getConstantInt(loc, op.getType(), cmpInfo.getGt());

  mlir::Value transformedResult;
  if (cmpInfo.getOrdering() != CmpOrdering::Partial) {
    // Total ordering
    mlir::Value lt =
        builder.createCompare(loc, CmpOpKind::lt, op.getLhs(), op.getRhs());
    mlir::Value selectOnLt = builder.createSelect(loc, lt, ltRes, gtRes);
    mlir::Value eq =
        builder.createCompare(loc, CmpOpKind::eq, op.getLhs(), op.getRhs());
    transformedResult = builder.createSelect(loc, eq, eqRes, selectOnLt);
  } else {
    // Partial ordering
    cir::ConstantOp unorderedRes = builder.getConstantInt(
        loc, op.getType(), cmpInfo.getUnordered().value());

    mlir::Value eq =
        builder.createCompare(loc, CmpOpKind::eq, op.getLhs(), op.getRhs());
    mlir::Value selectOnEq = builder.createSelect(loc, eq, eqRes, unorderedRes);
    mlir::Value gt =
        builder.createCompare(loc, CmpOpKind::gt, op.getLhs(), op.getRhs());
    mlir::Value selectOnGt = builder.createSelect(loc, gt, gtRes, selectOnEq);
    mlir::Value lt =
        builder.createCompare(loc, CmpOpKind::lt, op.getLhs(), op.getRhs());
    transformedResult = builder.createSelect(loc, lt, ltRes, selectOnGt);
  }

  op.replaceAllUsesWith(transformedResult);
  op.erase();
}

template <typename AttributeTy>
static llvm::SmallVector<mlir::Attribute>
prepareCtorDtorAttrList(mlir::MLIRContext *context,
                        llvm::ArrayRef<std::pair<std::string, uint32_t>> list) {
  llvm::SmallVector<mlir::Attribute> attrs;
  for (const auto &[name, priority] : list)
    attrs.push_back(AttributeTy::get(context, name, priority));
  return attrs;
}

void LoweringPreparePass::buildGlobalCtorDtorList() {
  if (!globalCtorList.empty()) {
    llvm::SmallVector<mlir::Attribute> globalCtors =
        prepareCtorDtorAttrList<cir::GlobalCtorAttr>(&getContext(),
                                                     globalCtorList);

    mlirModule->setAttr(cir::CIRDialect::getGlobalCtorsAttrName(),
                        mlir::ArrayAttr::get(&getContext(), globalCtors));
  }

  if (!globalDtorList.empty()) {
    llvm::SmallVector<mlir::Attribute> globalDtors =
        prepareCtorDtorAttrList<cir::GlobalDtorAttr>(&getContext(),
                                                     globalDtorList);
    mlirModule->setAttr(cir::CIRDialect::getGlobalDtorsAttrName(),
                        mlir::ArrayAttr::get(&getContext(), globalDtors));
  }
}

void LoweringPreparePass::buildCXXGlobalInitFunc() {
  if (dynamicInitializers.empty())
    return;

  // TODO: handle globals with a user-specified initialzation priority.
  // TODO: handle default priority more nicely.
  assert(!cir::MissingFeatures::opGlobalCtorPriority());

  SmallString<256> fnName;
  // Include the filename in the symbol name. Including "sub_" matches gcc
  // and makes sure these symbols appear lexicographically behind the symbols
  // with priority (TBD).  Module implementation units behave the same
  // way as a non-modular TU with imports.
  // TODO: check CXX20ModuleInits
  if (astCtx->getCurrentNamedModule() &&
      !astCtx->getCurrentNamedModule()->isModuleImplementation()) {
    llvm::raw_svector_ostream out(fnName);
    std::unique_ptr<clang::MangleContext> mangleCtx(
        astCtx->createMangleContext());
    cast<clang::ItaniumMangleContext>(*mangleCtx)
        .mangleModuleInitializer(astCtx->getCurrentNamedModule(), out);
  } else {
    fnName += "_GLOBAL__sub_I_";
    fnName += getTransformedFileName(mlirModule);
  }

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToEnd(&mlirModule.getBodyRegion().back());
  auto fnType = cir::FuncType::get({}, builder.getVoidTy());
  cir::FuncOp f =
      buildRuntimeFunction(builder, fnName, mlirModule.getLoc(), fnType,
                           cir::GlobalLinkageKind::ExternalLinkage);
  builder.setInsertionPointToStart(f.addEntryBlock());
  for (cir::FuncOp &f : dynamicInitializers)
    builder.createCallOp(f.getLoc(), f, {});
  // Add the global init function (not the individual ctor functions) to the
  // global ctor list.
  globalCtorList.emplace_back(fnName,
                              cir::GlobalCtorAttr::getDefaultPriority());

  cir::ReturnOp::create(builder, f.getLoc());
}

/// Lower a cir.array.ctor or cir.array.dtor into a do-while loop that
/// iterates over every element.  For cir.array.ctor ops whose partial_dtor
/// region is non-empty, the ctor loop is wrapped in a cir.cleanup.scope whose
/// EH cleanup performs a reverse destruction loop using the partial dtor body.
static void lowerArrayDtorCtorIntoLoop(cir::CIRBaseBuilderTy &builder,
                                       clang::ASTContext *astCtx,
                                       mlir::Operation *op, mlir::Type eltTy,
                                       mlir::Value addr,
                                       mlir::Value numElements,
                                       uint64_t arrayLen, bool isCtor) {
  mlir::Location loc = op->getLoc();
  bool isDynamic = numElements != nullptr;

  // TODO: instead of getting the size from the AST context, create alias for
  // PtrDiffTy and unify with CIRGen stuff.
  const unsigned sizeTypeSize =
      astCtx->getTypeSize(astCtx->getSignedSizeType());

  // Both constructors and destructors use end = begin + numElements.
  // Constructors iterate forward [begin, end).  Destructors iterate backward
  // from end, decrementing before calling the destructor on each element.
  mlir::Value begin, end;
  if (isDynamic) {
    begin = addr;
    end = cir::PtrStrideOp::create(builder, loc, eltTy, begin, numElements);
  } else {
    mlir::Value endOffsetVal =
        builder.getUnsignedInt(loc, arrayLen, sizeTypeSize);
    begin = cir::CastOp::create(builder, loc, eltTy,
                                cir::CastKind::array_to_ptrdecay, addr);
    end = cir::PtrStrideOp::create(builder, loc, eltTy, begin, endOffsetVal);
  }

  mlir::Value start = isCtor ? begin : end;
  mlir::Value stop = isCtor ? end : begin;

  // For dynamic destructors, guard against zero elements.
  // This places the destructor loop emitted below inside the if block.
  cir::IfOp ifOp;
  if (isDynamic) {
    mlir::Value guardCond;
    if (isCtor) {
      mlir::Value zero = builder.getUnsignedInt(loc, 0, sizeTypeSize);
      guardCond = cir::CmpOp::create(builder, loc, cir::CmpOpKind::ne,
                                     numElements, zero);
    } else {
      // We could check for numElements != 0 in this case too, but this matches
      // what classic codegen does.
      guardCond =
          cir::CmpOp::create(builder, loc, cir::CmpOpKind::ne, start, stop);
    }
    ifOp = cir::IfOp::create(builder, loc, guardCond,
                             /*withElseRegion=*/false,
                             [&](mlir::OpBuilder &, mlir::Location) {});
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }

  mlir::Value tmpAddr = builder.createAlloca(
      loc, /*addr type*/ builder.getPointerTo(eltTy),
      /*var type*/ eltTy, "__array_idx", builder.getAlignmentAttr(1));
  builder.createStore(loc, start, tmpAddr);

  mlir::Block *bodyBlock = &op->getRegion(0).front();

  // Clone the region body (ctor/dtor call and any setup ops like per-element
  // zero-init) into the loop, remapping the block argument to the current
  // element pointer.
  auto cloneRegionBodyInto = [&](mlir::Block *srcBlock,
                                 mlir::Value replacement) {
    mlir::IRMapping map;
    map.map(srcBlock->getArgument(0), replacement);
    for (mlir::Operation &regionOp : *srcBlock) {
      if (!mlir::isa<cir::YieldOp>(&regionOp))
        builder.clone(regionOp, map);
    }
  };

  mlir::Block *partialDtorBlock = nullptr;
  if (auto arrayCtor = mlir::dyn_cast<cir::ArrayCtor>(op)) {
    mlir::Region &partialDtor = arrayCtor.getPartialDtor();
    if (!partialDtor.empty())
      partialDtorBlock = &partialDtor.front();
  } else if (auto arrayDtor = mlir::dyn_cast<cir::ArrayDtor>(op)) {
    // When the element destructor may throw, reuse the body block as the
    // partial-dtor block so that an exception thrown by an element's dtor
    // continues the reverse-destruction loop in the EH cleanup region. The
    // body block already stores the next element pointer to `tmpAddr`
    // before invoking the dtor, so when an exception unwinds from the
    // dtor call `tmpAddr` already points at the element that threw, and
    // the cleanup loop picks up from `tmpAddr - 1` and walks back to
    // `begin`.
    if (arrayDtor.getDtorMayThrow())
      partialDtorBlock = bodyBlock;
  }

  auto emitCtorDtorLoop = [&]() {
    builder.createDoWhile(
        loc,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto currentElement = cir::LoadOp::create(b, loc, eltTy, tmpAddr);
          auto cmp = cir::CmpOp::create(builder, loc, cir::CmpOpKind::ne,
                                        currentElement, stop);
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto currentElement = cir::LoadOp::create(b, loc, eltTy, tmpAddr);
          if (isCtor) {
            cloneRegionBodyInto(bodyBlock, currentElement);
            mlir::Value stride = builder.getUnsignedInt(loc, 1, sizeTypeSize);
            auto nextElement = cir::PtrStrideOp::create(builder, loc, eltTy,
                                                        currentElement, stride);
            builder.createStore(loc, nextElement, tmpAddr);
          } else {
            mlir::Value stride = builder.getSignedInt(loc, -1, sizeTypeSize);
            auto prevElement = cir::PtrStrideOp::create(builder, loc, eltTy,
                                                        currentElement, stride);
            builder.createStore(loc, prevElement, tmpAddr);
            cloneRegionBodyInto(bodyBlock, prevElement);
          }

          cir::YieldOp::create(b, loc);
        });
  };

  if (partialDtorBlock) {
    cir::CleanupScopeOp::create(
        builder, loc, cir::CleanupKind::EH,
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          emitCtorDtorLoop();
          cir::YieldOp::create(b, loc);
        },
        /*cleanupBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto cur = cir::LoadOp::create(b, loc, eltTy, tmpAddr);
          auto cmp =
              cir::CmpOp::create(builder, loc, cir::CmpOpKind::ne, cur, begin);
          cir::IfOp::create(
              builder, loc, cmp, /*withElseRegion=*/false,
              [&](mlir::OpBuilder &b, mlir::Location loc) {
                builder.createDoWhile(
                    loc,
                    /*condBuilder=*/
                    [&](mlir::OpBuilder &b, mlir::Location loc) {
                      auto el = cir::LoadOp::create(b, loc, eltTy, tmpAddr);
                      auto neq = cir::CmpOp::create(
                          builder, loc, cir::CmpOpKind::ne, el, begin);
                      builder.createCondition(neq);
                    },
                    /*bodyBuilder=*/
                    [&](mlir::OpBuilder &b, mlir::Location loc) {
                      auto el = cir::LoadOp::create(b, loc, eltTy, tmpAddr);
                      mlir::Value negOne =
                          builder.getSignedInt(loc, -1, sizeTypeSize);
                      auto prev = cir::PtrStrideOp::create(builder, loc, eltTy,
                                                           el, negOne);
                      builder.createStore(loc, prev, tmpAddr);
                      cloneRegionBodyInto(partialDtorBlock, prev);
                      builder.createYield(loc);
                    });
                cir::YieldOp::create(builder, loc);
              });
          cir::YieldOp::create(b, loc);
        });
  } else {
    emitCtorDtorLoop();
  }

  if (ifOp)
    cir::YieldOp::create(builder, loc);

  op->erase();
}

void LoweringPreparePass::lowerArrayDtor(cir::ArrayDtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  mlir::Type eltTy = op->getRegion(0).getArgument(0).getType();

  if (op.getNumElements()) {
    lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(),
                               op.getNumElements(), /*arrayLen=*/0,
                               /*isCtor=*/false);
    return;
  }

  auto arrayLen =
      mlir::cast<cir::ArrayType>(op.getAddr().getType().getPointee()).getSize();
  lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(),
                             /*numElements=*/nullptr, arrayLen,
                             /*isCtor=*/false);
}

void LoweringPreparePass::lowerArrayCtor(cir::ArrayCtor op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  mlir::Type eltTy = op->getRegion(0).getArgument(0).getType();

  if (op.getNumElements()) {
    lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(),
                               op.getNumElements(), /*arrayLen=*/0,
                               /*isCtor=*/true);
    return;
  }

  auto arrayLen =
      mlir::cast<cir::ArrayType>(op.getAddr().getType().getPointee()).getSize();
  lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(),
                             /*numElements=*/nullptr, arrayLen,
                             /*isCtor=*/true);
}

cir::FuncOp LoweringPreparePass::getCalledFunction(cir::CallOp callOp) {
  mlir::SymbolRefAttr sym = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
      callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return symbolTables.lookupNearestSymbolFrom<cir::FuncOp>(callOp, sym);
}

void LoweringPreparePass::lowerTrivialCopyCall(cir::CallOp op) {
  cir::FuncOp funcOp = getCalledFunction(op);
  if (!funcOp)
    return;

  std::optional<cir::CtorKind> ctorKind = funcOp.getCxxConstructorKind();
  if (ctorKind && *ctorKind == cir::CtorKind::Copy &&
      funcOp.isCxxTrivialMemberFunction()) {
    // Replace the trivial copy constructor call with a `CopyOp`
    CIRBaseBuilderTy builder(getContext());
    mlir::ValueRange operands = op.getOperands();
    mlir::Value dest = operands[0];
    mlir::Value src = operands[1];
    builder.setInsertionPoint(op);
    builder.createCopy(dest, src);
    op.erase();
  }
}

cir::GlobalOp LoweringPreparePass::getOrCreateConstAggregateGlobal(
    CIRBaseBuilderTy &builder, mlir::Location loc, llvm::StringRef baseName,
    mlir::Type ty, mlir::TypedAttr constant) {
  // Look up (and lazily populate) the per-base-name cache.
  llvm::SmallVector<cir::GlobalOp, 1> &versions =
      constAggregateGlobals[baseName];

  // First, check globals we've already discovered for this base name.
  for (cir::GlobalOp gv : versions) {
    if (gv.getSymType() == ty && gv.getInitialValue() == constant)
      return gv;
  }

  // No cached match. Scan the module's symbol table starting from the next
  // unscanned version. In practice this should usually exit on the first
  // iteration, but it's possible that some other pass or a previous
  // invocation of this pass created globals using this same logic.
  llvm::SmallString<128> name(baseName);
  size_t baseLen = name.size();
  unsigned version = versions.size();
  while (true) {
    name.resize(baseLen);
    if (version != 0) {
      name.push_back('.');
      llvm::Twine(version).toVector(name);
    }
    auto existingGv = symbolTables.lookupSymbolIn<cir::GlobalOp>(
        mlirModule, mlir::StringAttr::get(&getContext(), name));
    if (!existingGv)
      break;
    versions.push_back(existingGv);
    if (existingGv.getSymType() == ty &&
        existingGv.getInitialValue() == constant)
      return existingGv;
    ++version;
  }

  // No match found, create a new global. The loop above found an unused name.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(mlirModule.getBody());
  auto gv =
      cir::GlobalOp::create(builder, loc, name, ty,
                            /*isConstant=*/true,
                            cir::LangAddressSpaceAttr::get(
                                &getContext(), cir::LangAddressSpace::Default),
                            cir::GlobalLinkageKind::PrivateLinkage);
  mlir::SymbolTable::setSymbolVisibility(
      gv, mlir::SymbolTable::Visibility::Private);
  gv.setInitialValueAttr(constant);

  // Keep the cached symbol table in sync with the new global so subsequent
  // lookups for other base names find it.
  symbolTables.getSymbolTable(mlirModule).insert(gv);

  versions.push_back(gv);
  return gv;
}

void LoweringPreparePass::lowerStoreOfConstAggregate(cir::StoreOp op) {
  // Check if the value operand is a cir.const with aggregate type.
  auto constOp = op.getValue().getDefiningOp<cir::ConstantOp>();
  if (!constOp)
    return;

  mlir::Type ty = constOp.getType();
  if (!mlir::isa<cir::ArrayType, cir::RecordType>(ty))
    return;

  // Only transform stores to local variables (backed by cir.alloca).
  // Stores to other addresses (e.g. base_class_addr) should not be
  // transformed as they may be partial initializations.
  auto alloca = op.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!alloca)
    return;

  mlir::TypedAttr constant = constOp.getValue();

  // OG implements several optimization tiers for constant aggregate
  // initialization. For now we always create a global constant + memcpy
  // (shouldCreateMemCpyFromGlobal). Future work can add the intermediate
  // tiers.
  assert(!cir::MissingFeatures::shouldUseBZeroPlusStoresToInitialize());
  assert(!cir::MissingFeatures::shouldUseMemSetToInitialize());
  assert(!cir::MissingFeatures::shouldSplitConstantStore());

  // Get function name from parent cir.func.
  auto func = op->getParentOfType<cir::FuncOp>();
  if (!func)
    return;
  llvm::StringRef funcName = func.getSymName();

  // Get variable name from the alloca.
  llvm::StringRef varName = alloca.getName();

  // Build base name: __const.<func>.<var>
  std::string baseName = ("__const." + funcName + "." + varName).str();
  CIRBaseBuilderTy builder(getContext());

  // Check for existing globals and create a new global with a unique name
  // if no match is found.
  cir::GlobalOp gv = getOrCreateConstAggregateGlobal(builder, op.getLoc(),
                                                     baseName, ty, constant);

  // Now replace the store with get_global + copy.
  builder.setInsertionPoint(op);

  auto ptrTy = cir::PointerType::get(ty);
  mlir::Value globalPtr =
      cir::GetGlobalOp::create(builder, op.getLoc(), ptrTy, gv.getSymName());

  // Replace store with copy.
  builder.createCopy(op.getAddr(), globalPtr);

  // Erase the original store.
  op.erase();

  // Erase the cir.const if it has no remaining users.
  if (constOp.use_empty())
    constOp.erase();
}

void LoweringPreparePass::runOnOp(mlir::Operation *op) {
  if (auto arrayCtor = dyn_cast<cir::ArrayCtor>(op)) {
    lowerArrayCtor(arrayCtor);
  } else if (auto arrayDtor = dyn_cast<cir::ArrayDtor>(op)) {
    lowerArrayDtor(arrayDtor);
  } else if (auto cast = mlir::dyn_cast<cir::CastOp>(op)) {
    lowerCastOp(cast);
  } else if (auto complexDiv = mlir::dyn_cast<cir::ComplexDivOp>(op)) {
    lowerComplexDivOp(complexDiv);
  } else if (auto complexMul = mlir::dyn_cast<cir::ComplexMulOp>(op)) {
    lowerComplexMulOp(complexMul);
  } else if (auto glob = mlir::dyn_cast<cir::GlobalOp>(op)) {
    lowerGlobalOp(glob);
  } else if (auto getGlob = mlir::dyn_cast<cir::GetGlobalOp>(op)) {
    lowerGetGlobalOp(getGlob);
  } else if (auto unaryOp = mlir::dyn_cast<cir::UnaryOpInterface>(op)) {
    lowerUnaryOp(unaryOp);
  } else if (auto callOp = dyn_cast<cir::CallOp>(op)) {
    lowerTrivialCopyCall(callOp);
  } else if (auto storeOp = dyn_cast<cir::StoreOp>(op)) {
    lowerStoreOfConstAggregate(storeOp);
  } else if (auto fnOp = dyn_cast<cir::FuncOp>(op)) {
    if (auto globalCtor = fnOp.getGlobalCtorPriority())
      globalCtorList.emplace_back(fnOp.getName(), globalCtor.value());
    else if (auto globalDtor = fnOp.getGlobalDtorPriority())
      globalDtorList.emplace_back(fnOp.getName(), globalDtor.value());

    if (mlir::Attribute attr =
            fnOp->getAttr(cir::CUDAKernelNameAttr::getMnemonic())) {
      auto kernelNameAttr = dyn_cast<CUDAKernelNameAttr>(attr);
      llvm::StringRef kernelName = kernelNameAttr.getKernelName();
      cudaKernelMap[kernelName] = fnOp;
    }
  } else if (auto threeWayCmp = dyn_cast<cir::CmpThreeWayOp>(op)) {
    lowerThreeWayCmpOp(threeWayCmp);
  } else if (auto initOp = dyn_cast<cir::LocalInitOp>(op)) {
    lowerLocalInitOp(initOp);
  }
}

static llvm::StringRef getCUDAPrefix(clang::ASTContext *astCtx) {
  if (astCtx->getLangOpts().HIP)
    return "hip";
  return "cuda";
}

static std::string addUnderscoredPrefix(llvm::StringRef prefix,
                                        llvm::StringRef name) {
  return ("__" + prefix + name).str();
}

/// Creates a global constructor function for the module:
///
/// For CUDA:
/// \code
/// void __cuda_module_ctor() {
///     Handle = __cudaRegisterFatBinary(GpuBinaryBlob);
///     __cuda_register_globals(Handle);
/// }
/// \endcode
///
/// For HIP:
/// \code
/// void __hip_module_ctor() {
///     if (__hip_gpubin_handle == 0) {
///         __hip_gpubin_handle  = __hipRegisterFatBinary(GpuBinaryBlob);
///         __hip_register_globals(__hip_gpubin_handle);
///     }
/// }
/// \endcode
void LoweringPreparePass::buildCUDAModuleCtor() {
  bool isHIP = astCtx->getLangOpts().HIP;

  if (astCtx->getLangOpts().GPURelocatableDeviceCode)
    llvm_unreachable("GPU RDC NYI");

  // For CUDA without -fgpu-rdc, it's safe to stop generating ctor
  // if there's nothing to register.
  if (cudaKernelMap.empty())
    return;

  // There's no device-side binary, so no need to proceed for CUDA.
  // HIP has to create an external symbol in this case, which is NYI.
  mlir::Attribute cudaBinaryHandleAttr =
      mlirModule->getAttr(CIRDialect::getCUDABinaryHandleAttrName());
  if (!cudaBinaryHandleAttr) {
    if (isHIP)
      assert(!cir::MissingFeatures::hipModuleCtor());
    return;
  }

  llvm::StringRef cudaGPUBinaryName =
      mlir::cast<CUDABinaryHandleAttr>(cudaBinaryHandleAttr)
          .getName()
          .getValue();

  llvm::vfs::FileSystem &vfs =
      astCtx->getSourceManager().getFileManager().getVirtualFileSystem();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> gpuBinaryOrErr =
      vfs.getBufferForFile(cudaGPUBinaryName);
  if (std::error_code ec = gpuBinaryOrErr.getError()) {
    mlirModule->emitError("cannot open GPU binary file: " + cudaGPUBinaryName +
                          ": " + ec.message());
    return;
  }
  std::unique_ptr<llvm::MemoryBuffer> gpuBinary =
      std::move(gpuBinaryOrErr.get());

  // Set up common types and builder.
  llvm::StringRef cudaPrefix = getCUDAPrefix(astCtx);
  mlir::Location loc = mlirModule->getLoc();
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  Type voidTy = builder.getVoidTy();
  PointerType voidPtrTy = builder.getVoidPtrTy();
  PointerType voidPtrPtrTy = builder.getPointerTo(voidPtrTy);
  IntType intTy = builder.getSIntNTy(32);
  IntType charTy = cir::IntType::get(&getContext(), astCtx->getCharWidth(),
                                     /*isSigned=*/false);

  // --- Create fatbin globals ---

  // The section names are different for MAC OS X.
  llvm::StringRef fatbinConstName =
      astCtx->getLangOpts().HIP ? ".hip_fatbin" : ".nv_fatbin";

  llvm::StringRef fatbinSectionName =
      astCtx->getLangOpts().HIP ? ".hipFatBinSegment" : ".nvFatBinSegment";

  // Create the fatbin string constant with GPU binary contents.
  auto fatbinType =
      ArrayType::get(&getContext(), charTy, gpuBinary->getBuffer().size());
  std::string fatbinStrName = addUnderscoredPrefix(cudaPrefix, "_fatbin_str");
  GlobalOp fatbinStr = GlobalOp::create(builder, loc, fatbinStrName, fatbinType,
                                        /*isConstant=*/true, {},
                                        GlobalLinkageKind::PrivateLinkage);
  fatbinStr.setAlignment(8);
  fatbinStr.setInitialValueAttr(cir::ConstArrayAttr::get(
      fatbinType, StringAttr::get(gpuBinary->getBuffer(), fatbinType)));
  fatbinStr.setSection(fatbinConstName);
  fatbinStr.setPrivate();

  // Create the fatbin wrapper struct:
  //    struct { int magic; int version; void *fatbin; void *unused; };
  auto fatbinWrapperType = RecordType::get(
      &getContext(), {intTy, intTy, voidPtrTy, voidPtrTy},
      /*packed=*/false, /*padded=*/false, RecordType::RecordKind::Struct);
  std::string fatbinWrapperName =
      addUnderscoredPrefix(cudaPrefix, "_fatbin_wrapper");
  GlobalOp fatbinWrapper = GlobalOp::create(
      builder, loc, fatbinWrapperName, fatbinWrapperType,
      /*isConstant=*/true, {}, GlobalLinkageKind::PrivateLinkage);
  fatbinWrapper.setSection(fatbinSectionName);

  constexpr unsigned cudaFatMagic = 0x466243b1;
  constexpr unsigned hipFatMagic = 0x48495046;
  unsigned fatMagic = isHIP ? hipFatMagic : cudaFatMagic;

  auto magicInit = IntAttr::get(intTy, fatMagic);
  auto versionInit = IntAttr::get(intTy, 1);
  auto fatbinStrSymbol =
      mlir::FlatSymbolRefAttr::get(fatbinStr.getSymNameAttr());
  auto fatbinInit = GlobalViewAttr::get(voidPtrTy, fatbinStrSymbol);
  mlir::TypedAttr unusedInit = builder.getConstNullPtrAttr(voidPtrTy);
  fatbinWrapper.setInitialValueAttr(cir::ConstRecordAttr::get(
      fatbinWrapperType,
      mlir::ArrayAttr::get(&getContext(),
                           {magicInit, versionInit, fatbinInit, unusedInit})));

  // Create the GPU binary handle global variable.
  std::string gpubinHandleName =
      addUnderscoredPrefix(cudaPrefix, "_gpubin_handle");

  GlobalOp gpuBinHandle = GlobalOp::create(
      builder, loc, gpubinHandleName, voidPtrPtrTy,
      /*isConstant=*/false, {}, cir::GlobalLinkageKind::InternalLinkage);
  gpuBinHandle.setInitialValueAttr(builder.getConstNullPtrAttr(voidPtrPtrTy));
  gpuBinHandle.setPrivate();

  // Declare this function:
  //    void **__{cuda|hip}RegisterFatBinary(void *);

  std::string regFuncName =
      addUnderscoredPrefix(cudaPrefix, "RegisterFatBinary");
  FuncType regFuncType = FuncType::get({voidPtrTy}, voidPtrPtrTy);
  cir::FuncOp regFunc =
      buildRuntimeFunction(builder, regFuncName, loc, regFuncType);

  std::string moduleCtorName = addUnderscoredPrefix(cudaPrefix, "_module_ctor");
  cir::FuncOp moduleCtor = buildRuntimeFunction(
      builder, moduleCtorName, loc, FuncType::get({}, voidTy),
      GlobalLinkageKind::InternalLinkage);

  globalCtorList.emplace_back(moduleCtorName,
                              cir::GlobalCtorAttr::getDefaultPriority());
  builder.setInsertionPointToStart(moduleCtor.addEntryBlock());
  assert(!cir::MissingFeatures::opGlobalCtorPriority());
  if (isHIP) {
    // --- Create HIP CTOR ---
    //   if (__hip_gpubin_handle == nullptr)
    //     __hip_gpubin_handle = __hipRegisterFatBinary(&fatbinWrapper);
    //   __hip_register_globals(__hip_gpubin_handle);
    //   atexit(__hip_module_dtor);
    mlir::Block *entryBlock = builder.getInsertionBlock();
    mlir::Region *parent = entryBlock->getParent();
    mlir::Block *ifBlock = builder.createBlock(parent);
    mlir::Block *exitBlock = builder.createBlock(parent);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(entryBlock);
      mlir::Value handle =
          builder.createLoad(loc, builder.createGetGlobal(gpuBinHandle));
      auto handlePtrTy = mlir::cast<cir::PointerType>(handle.getType());
      mlir::Value nullPtr = builder.getNullPtr(handlePtrTy, loc);
      mlir::Value isNull =
          builder.createCompare(loc, cir::CmpOpKind::eq, handle, nullPtr);
      cir::BrCondOp::create(builder, loc, isNull, ifBlock, exitBlock);
    }
    {
      // Handle is null: load the fatbin and register it.
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifBlock);
      mlir::Value wrapper = builder.createGetGlobal(fatbinWrapper);
      mlir::Value fatbinVoidPtr = builder.createBitcast(wrapper, voidPtrTy);
      cir::CallOp gpuBinaryHandleCall =
          builder.createCallOp(loc, regFunc, fatbinVoidPtr);
      mlir::Value gpuBinaryHandle = gpuBinaryHandleCall.getResult();
      // Store the value back to the global `__hip_gpubin_handle`.
      mlir::Value gpuBinaryHandleGlobal = builder.createGetGlobal(gpuBinHandle);
      builder.createStore(loc, gpuBinaryHandle, gpuBinaryHandleGlobal);
      cir::BrOp::create(builder, loc, exitBlock);
    }
    {
      // Exit block: load the (possibly newly-registered) handle, call
      // __hip_register_globals, and register the module dtor with atexit().
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(exitBlock);
      mlir::Value gHandle =
          builder.createLoad(loc, builder.createGetGlobal(gpuBinHandle));

      if (std::optional<FuncOp> regGlobal = buildCUDARegisterGlobals())
        builder.createCallOp(loc, *regGlobal, gHandle);

      if (std::optional<FuncOp> dtor = buildHIPModuleDtor()) {
        cir::CIRBaseBuilderTy globalBuilder(getContext());
        globalBuilder.setInsertionPointToStart(mlirModule.getBody());
        FuncOp atexit = buildRuntimeFunction(
            globalBuilder, "atexit", loc,
            FuncType::get(PointerType::get(dtor->getFunctionType()), intTy));
        mlir::Value dtorFunc = GetGlobalOp::create(
            builder, loc, PointerType::get(dtor->getFunctionType()),
            mlir::FlatSymbolRefAttr::get(dtor->getSymNameAttr()));
        builder.createCallOp(loc, atexit, dtorFunc);
      }
      cir::ReturnOp::create(builder, loc);
    }
    return;
  }
  if (!astCtx->getLangOpts().GPURelocatableDeviceCode) {

    // --- Create CUDA CTOR-DTOR ---
    // Register binary with CUDA runtime. This is substantially different in
    // default mode vs. separate compilation.
    // Corresponding code:
    //     gpuBinaryHandle = __cudaRegisterFatBinary(&fatbinWrapper);
    mlir::Value wrapper = builder.createGetGlobal(fatbinWrapper);
    mlir::Value fatbinVoidPtr = builder.createBitcast(wrapper, voidPtrTy);
    cir::CallOp gpuBinaryHandleCall =
        builder.createCallOp(loc, regFunc, fatbinVoidPtr);
    mlir::Value gpuBinaryHandle = gpuBinaryHandleCall.getResult();
    // Store the value back to the global `__cuda_gpubin_handle`.
    mlir::Value gpuBinaryHandleGlobal = builder.createGetGlobal(gpuBinHandle);
    builder.createStore(loc, gpuBinaryHandle, gpuBinaryHandleGlobal);

    // --- Generate __cuda_register_globals and call it ---
    if (std::optional<FuncOp> regGlobal = buildCUDARegisterGlobals()) {
      builder.createCallOp(loc, *regGlobal, gpuBinaryHandle);
    }

    // From CUDA 10.1 onwards, we must call this function to end registration:
    //      void __cudaRegisterFatBinaryEnd(void **fatbinHandle);
    // This is CUDA-specific, so no need to use `addUnderscoredPrefix`.
    if (clang::CudaFeatureEnabled(
            astCtx->getTargetInfo().getSDKVersion(),
            clang::CudaFeature::CUDA_USES_FATBIN_REGISTER_END)) {
      cir::CIRBaseBuilderTy globalBuilder(getContext());
      globalBuilder.setInsertionPointToStart(mlirModule.getBody());
      FuncOp endFunc =
          buildRuntimeFunction(globalBuilder, "__cudaRegisterFatBinaryEnd", loc,
                               FuncType::get({voidPtrPtrTy}, voidTy));
      builder.createCallOp(loc, endFunc, gpuBinaryHandle);
    }
  } else
    llvm_unreachable("GPU RDC NYI");

  // Create destructor and register it with atexit() the way NVCC does it. Doing
  // it during regular destructor phase worked in CUDA before 9.2 but results in
  // double-free in 9.2.
  if (std::optional<FuncOp> dtor = buildCUDAModuleDtor()) {

    // extern "C" int atexit(void (*f)(void));
    cir::CIRBaseBuilderTy globalBuilder(getContext());
    globalBuilder.setInsertionPointToStart(mlirModule.getBody());
    FuncOp atexit = buildRuntimeFunction(
        globalBuilder, "atexit", loc,
        FuncType::get(PointerType::get(dtor->getFunctionType()), intTy));
    mlir::Value dtorFunc = GetGlobalOp::create(
        builder, loc, PointerType::get(dtor->getFunctionType()),
        mlir::FlatSymbolRefAttr::get(dtor->getSymNameAttr()));
    builder.createCallOp(loc, atexit, dtorFunc);
  }
  cir::ReturnOp::create(builder, loc);
}

std::optional<FuncOp> LoweringPreparePass::buildCUDAModuleDtor() {
  if (!mlirModule->getAttr(CIRDialect::getCUDABinaryHandleAttrName()))
    return {};

  llvm::StringRef prefix = getCUDAPrefix(astCtx);

  VoidType voidTy = VoidType::get(&getContext());
  PointerType voidPtrPtrTy = PointerType::get(PointerType::get(voidTy));

  mlir::Location loc = mlirModule.getLoc();

  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  // define: void __cudaUnregisterFatBinary(void ** handle);
  std::string unregisterFuncName =
      addUnderscoredPrefix(prefix, "UnregisterFatBinary");
  FuncOp unregisterFunc = buildRuntimeFunction(
      builder, unregisterFuncName, loc, FuncType::get({voidPtrPtrTy}, voidTy));

  // void __cuda_module_dtor();
  // Despite the name, OG doesn't treat it as a destructor, so it shouldn't be
  // put into globalDtorList. If it were a real dtor, then it would cause
  // double free above CUDA 9.2. The way to use it is to manually call
  // atexit() at end of module ctor.
  std::string dtorName = addUnderscoredPrefix(prefix, "_module_dtor");
  FuncOp dtor =
      buildRuntimeFunction(builder, dtorName, loc, FuncType::get({}, voidTy),
                           GlobalLinkageKind::InternalLinkage);

  builder.setInsertionPointToStart(dtor.addEntryBlock());

  // For dtor, we only need to call:
  //    __cudaUnregisterFatBinary(__cuda_gpubin_handle);

  std::string gpubinName = addUnderscoredPrefix(prefix, "_gpubin_handle");
  GlobalOp gpubinGlobal = cast<GlobalOp>(mlirModule.lookupSymbol(gpubinName));
  mlir::Value gpubinAddress = builder.createGetGlobal(gpubinGlobal);
  mlir::Value gpubin = builder.createLoad(loc, gpubinAddress);
  builder.createCallOp(loc, unregisterFunc, gpubin);
  ReturnOp::create(builder, loc);

  return dtor;
}

/// Build the HIP module dtor:
///
///     void __hip_module_dtor() {
///       if (__hip_gpubin_handle != nullptr) {
///         __hipUnregisterFatBinary(__hip_gpubin_handle);
///         __hip_gpubin_handle = nullptr;
///       }
///     }
///
/// Despite the name, OG doesn't treat this as a real destructor: putting it on
/// the dtor list would cause a double-free. It is meant to be registered via
/// atexit() at the end of the module ctor.
std::optional<FuncOp> LoweringPreparePass::buildHIPModuleDtor() {
  if (!mlirModule->getAttr(CIRDialect::getCUDABinaryHandleAttrName()))
    return {};

  llvm::StringRef prefix = getCUDAPrefix(astCtx);

  VoidType voidTy = VoidType::get(&getContext());
  PointerType voidPtrPtrTy = PointerType::get(PointerType::get(voidTy));

  mlir::Location loc = mlirModule.getLoc();

  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  // void __hipUnregisterFatBinary(void ** handle);
  std::string unregisterFuncName =
      addUnderscoredPrefix(prefix, "UnregisterFatBinary");
  FuncOp unregisterFunc = buildRuntimeFunction(
      builder, unregisterFuncName, loc, FuncType::get({voidPtrPtrTy}, voidTy));

  std::string dtorName = addUnderscoredPrefix(prefix, "_module_dtor");
  FuncOp dtor =
      buildRuntimeFunction(builder, dtorName, loc, FuncType::get({}, voidTy),
                           GlobalLinkageKind::InternalLinkage);

  std::string gpubinName = addUnderscoredPrefix(prefix, "_gpubin_handle");
  GlobalOp gpuBinGlobal = cast<GlobalOp>(mlirModule.lookupSymbol(gpubinName));

  mlir::Block *entryBlock = dtor.addEntryBlock();
  mlir::Block *ifBlock = builder.createBlock(&dtor.getBody());
  mlir::Block *exitBlock = builder.createBlock(&dtor.getBody());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(entryBlock);
  mlir::Value handle =
      builder.createLoad(loc, builder.createGetGlobal(gpuBinGlobal));
  auto handlePtrTy = mlir::cast<cir::PointerType>(handle.getType());
  mlir::Value nullPtr = builder.getNullPtr(handlePtrTy, loc);
  mlir::Value isNotNull =
      builder.createCompare(loc, cir::CmpOpKind::ne, handle, nullPtr);
  cir::BrCondOp::create(builder, loc, isNotNull, ifBlock, exitBlock);

  {
    // Handle is non-null: unregister and clear it.
    mlir::OpBuilder::InsertionGuard ifGuard(builder);
    builder.setInsertionPointToStart(ifBlock);
    builder.createCallOp(loc, unregisterFunc, handle);
    builder.createStore(loc, nullPtr, builder.createGetGlobal(gpuBinGlobal));
    cir::BrOp::create(builder, loc, exitBlock);
  }
  {
    mlir::OpBuilder::InsertionGuard exitGuard(builder);
    builder.setInsertionPointToStart(exitBlock);
    cir::ReturnOp::create(builder, loc);
  }

  return dtor;
}

std::optional<FuncOp> LoweringPreparePass::buildCUDARegisterGlobals() {
  // There is nothing to register.
  if (cudaKernelMap.empty())
    return {};

  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  mlir::Location loc = mlirModule.getLoc();
  llvm::StringRef cudaPrefix = getCUDAPrefix(astCtx);

  auto voidTy = VoidType::get(&getContext());
  auto voidPtrTy = PointerType::get(voidTy);
  auto voidPtrPtrTy = PointerType::get(voidPtrTy);

  // Create the function:
  //      void __cuda_register_globals(void **fatbinHandle)
  std::string regGlobalFuncName =
      addUnderscoredPrefix(cudaPrefix, "_register_globals");
  auto regGlobalFuncTy = FuncType::get({voidPtrPtrTy}, voidTy);
  FuncOp regGlobalFunc =
      buildRuntimeFunction(builder, regGlobalFuncName, loc, regGlobalFuncTy,
                           /*linkage=*/GlobalLinkageKind::InternalLinkage);
  builder.setInsertionPointToStart(regGlobalFunc.addEntryBlock());

  buildCUDARegisterGlobalFunctions(builder, regGlobalFunc);
  // TODO: Handle shadow registration
  assert(!cir::MissingFeatures::globalRegistration());

  ReturnOp::create(builder, loc);
  return regGlobalFunc;
}

void LoweringPreparePass::buildCUDARegisterGlobalFunctions(
    cir::CIRBaseBuilderTy &builder, FuncOp regGlobalFunc) {
  mlir::Location loc = mlirModule.getLoc();
  llvm::StringRef cudaPrefix = getCUDAPrefix(astCtx);
  cir::CIRDataLayout dataLayout(mlirModule);

  auto voidTy = VoidType::get(&getContext());
  auto voidPtrTy = PointerType::get(voidTy);
  auto voidPtrPtrTy = PointerType::get(voidPtrTy);
  IntType intTy = builder.getSIntNTy(32);
  IntType charTy = cir::IntType::get(&getContext(), astCtx->getCharWidth(),
                                     /*isSigned=*/false);

  // Extract the GPU binary handle argument.
  mlir::Value fatbinHandle = *regGlobalFunc.args_begin();

  cir::CIRBaseBuilderTy globalBuilder(getContext());
  globalBuilder.setInsertionPointToStart(mlirModule.getBody());

  // Declare CUDA internal functions:
  // int __cudaRegisterFunction(
  //   void **fatbinHandle,
  //   const char *hostFunc,
  //   char *deviceFunc,
  //   const char *deviceName,
  //   int threadLimit,
  //   uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
  //   int *wsize
  // )
  // OG doesn't care about the types at all. They're treated as void*.

  FuncOp cudaRegisterFunction = buildRuntimeFunction(
      globalBuilder, addUnderscoredPrefix(cudaPrefix, "RegisterFunction"), loc,
      FuncType::get({voidPtrPtrTy, voidPtrTy, voidPtrTy, voidPtrTy, intTy,
                     voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy},
                    intTy));

  auto makeConstantString = [&](llvm::StringRef str) -> GlobalOp {
    auto strType = ArrayType::get(&getContext(), charTy, 1 + str.size());
    auto tmpString = cir::GlobalOp::create(
        globalBuilder, loc, (".str" + str).str(), strType,
        /*isConstant=*/true, {},
        /*linkage=*/cir::GlobalLinkageKind::PrivateLinkage);

    // We must make the string zero-terminated.
    tmpString.setInitialValueAttr(
        ConstArrayAttr::get(strType, StringAttr::get(str + "\0", strType)));
    tmpString.setPrivate();
    return tmpString;
  };

  cir::ConstantOp cirNullPtr = builder.getNullPtr(voidPtrTy, loc);
  bool isHIP = astCtx->getLangOpts().HIP;
  for (auto kernelName : cudaKernelMap.keys()) {
    FuncOp deviceStub = cudaKernelMap[kernelName];
    GlobalOp deviceFuncStr = makeConstantString(kernelName);
    mlir::Value deviceFunc = builder.createBitcast(
        builder.createGetGlobal(deviceFuncStr), voidPtrTy);

    mlir::Value hostFunc;
    if (isHIP) {
      // Under HIP, the kernel-handle is a GlobalOp shadow created by CIR
      // codegen and named with the kernel-reference mangled name (e.g.
      // `@_Z2fnv` pointing at the device-stub function
      // `_Z17__device_stub__fnv`). The CUDAKernelNameAttr on the device-stub
      // uses the same name, so we can resolve the shadow by symbol lookup.
      auto funcHandle = cast<GlobalOp>(mlirModule.lookupSymbol(kernelName));
      hostFunc =
          builder.createBitcast(builder.createGetGlobal(funcHandle), voidPtrTy);
    } else {
      hostFunc = builder.createBitcast(
          GetGlobalOp::create(
              builder, loc, PointerType::get(deviceStub.getFunctionType()),
              mlir::FlatSymbolRefAttr::get(deviceStub.getSymNameAttr())),
          voidPtrTy);
    }
    builder.createCallOp(
        loc, cudaRegisterFunction,
        {fatbinHandle, hostFunc, deviceFunc, deviceFunc,
         ConstantOp::create(builder, loc, IntAttr::get(intTy, -1)), cirNullPtr,
         cirNullPtr, cirNullPtr, cirNullPtr, cirNullPtr});
  }
}

void LoweringPreparePass::runOnOperation() {
  mlir::Operation *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    mlirModule = cast<::mlir::ModuleOp>(op);

  llvm::SmallVector<mlir::Operation *> opsToTransform;

  op->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::ArrayCtor, cir::ArrayDtor, cir::CastOp,
                  cir::ComplexMulOp, cir::ComplexDivOp, cir::DynamicCastOp,
                  cir::FuncOp, cir::CallOp, cir::GetGlobalOp, cir::GlobalOp,
                  cir::StoreOp, cir::CmpThreeWayOp, cir::IncOp, cir::DecOp,
                  cir::MinusOp, cir::NotOp, cir::LocalInitOp>(op))
      opsToTransform.push_back(op);
  });

  for (mlir::Operation *o : opsToTransform)
    runOnOp(o);

  buildCXXGlobalInitFunc();
  if (astCtx->getLangOpts().CUDA && !astCtx->getLangOpts().CUDAIsDevice)
    buildCUDAModuleCtor();

  buildGlobalCtorDtorList();
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}

std::unique_ptr<Pass>
mlir::createLoweringPreparePass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LoweringPreparePass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
