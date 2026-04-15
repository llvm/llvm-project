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

/// Return the FuncOp called by `callOp`.
static cir::FuncOp getCalledFunction(cir::CallOp callOp) {
  mlir::SymbolRefAttr sym = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
      callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<cir::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

namespace {
struct LoweringPreparePass
    : public impl::LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  void runOnOp(mlir::Operation *op);
  void lowerCastOp(cir::CastOp op);
  void lowerComplexDivOp(cir::ComplexDivOp op);
  void lowerComplexMulOp(cir::ComplexMulOp op);
  void lowerUnaryOp(cir::UnaryOpInterface op);
  void lowerGlobalOp(cir::GlobalOp op);
  void lowerThreeWayCmpOp(cir::CmpThreeWayOp op);
  void lowerArrayDtor(cir::ArrayDtor op);
  void lowerArrayCtor(cir::ArrayCtor op);
  void lowerTrivialCopyCall(cir::CallOp op);
  void lowerStoreOfConstAggregate(cir::StoreOp op);

  /// Build the function that initializes the specified global
  cir::FuncOp buildCXXGlobalVarDeclInitFunc(cir::GlobalOp op);

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

  cir::GlobalOp buildRuntimeVariable(
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
  std::optional<FuncOp> buildCUDARegisterGlobals();
  void buildCUDARegisterGlobalFunctions(cir::CIRBaseBuilderTy &builder,
                                        FuncOp regGlobalFunc);

  /// Handle static local variable initialization with guard variables.
  void handleStaticLocal(cir::GlobalOp globalOp, cir::GetGlobalOp getGlobalOp);

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
        globalOp->emitError("NYI: guard COMDAT for weak linkage");
        return {};
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

  /// Tracks existing dynamic initializers.
  llvm::StringMap<uint32_t> dynamicInitializerNames;
  llvm::SmallVector<cir::FuncOp> dynamicInitializers;

  /// Tracks guard variables for static locals (keyed by global symbol name).
  llvm::StringMap<cir::GlobalOp> staticLocalDeclGuardMap;

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

  /// Emit the guarded initialization for a static local variable.
  /// This handles the if/else structure after the guard byte check,
  /// following OG's ItaniumCXXABI::EmitGuardedInit skeleton.
  void emitCXXGuardedInitIf(CIRBaseBuilderTy &builder, cir::GlobalOp globalOp,
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

      // Emit the initializer and add a global destructor if appropriate.
      auto &ctorRegion = globalOp.getCtorRegion();
      assert(!ctorRegion.empty() && "This should never be empty here.");
      if (!ctorRegion.hasOneBlock())
        llvm_unreachable("Multiple blocks NYI");
      mlir::Block &block = ctorRegion.front();
      mlir::Block *insertBlock = builder.getInsertionBlock();
      insertBlock->getOperations().splice(insertBlock->end(),
                                          block.getOperations(), block.begin(),
                                          std::prev(block.end()));
      builder.setInsertionPointToEnd(insertBlock);
      ctorRegion.getBlocks().clear();

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
      // For local variables, store 1 into the first byte of the guard variable
      // after the object initialization completes so that initialization is
      // retried if initialization is interrupted by an exception.
      globalOp->emitError("NYI: non-threadsafe init for local variables");
      return;
    }

    builder.createYield(loc); // Outermost IfOp
  }

  void setASTContext(clang::ASTContext *c) { astCtx = c; }
};

} // namespace

cir::GlobalOp LoweringPreparePass::buildRuntimeVariable(
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

  // Move over the initialzation code of the ctor region.
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

    // Create a variable that binds the atexit to this shared object.
    builder.setInsertionPointToStart(&mlirModule.getBodyRegion().front());
    cir::GlobalOp handle = buildRuntimeVariable(
        builder, "__dso_handle", op.getLoc(), builder.getI8Type(),
        cir::GlobalLinkageKind::ExternalLinkage, cir::VisibilityKind::Hidden);

    // If this is a simple call to a destructor, get the called function.
    // Otherwise, create a helper function for the entire dtor region,
    // replacing the current dtor region body with a call to the helper
    // function.
    cir::CallOp dtorCall;
    cir::FuncOp dtorFunc =
        getOrCreateDtorFunc(builder, op, dtorRegion, dtorCall);

    // Create a runtime helper function:
    //    extern "C" int __cxa_atexit(void (*f)(void *), void *p, void *d);
    auto voidPtrTy = cir::PointerType::get(voidTy);
    auto voidFnTy = cir::FuncType::get({voidPtrTy}, voidTy);
    auto voidFnPtrTy = cir::PointerType::get(voidFnTy);
    auto handlePtrTy = cir::PointerType::get(handle.getSymType());
    auto fnAtExitType =
        cir::FuncType::get({voidFnPtrTy, voidPtrTy, handlePtrTy}, voidTy);
    const char *nameAtExit = "__cxa_atexit";
    cir::FuncOp fnAtExit =
        buildRuntimeFunction(builder, nameAtExit, op.getLoc(), fnAtExitType);

    // Replace the dtor (or helper) call with a call to
    //   __cxa_atexit(&dtor, &var, &__dso_handle)
    builder.setInsertionPointAfter(dtorCall);
    mlir::Value args[3];
    auto dtorPtrTy = cir::PointerType::get(dtorFunc.getFunctionType());
    // dtorPtrTy
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
    entryBB->getOperations().splice(entryBB->end(), dtorBlock.getOperations(),
                                    dtorBlock.begin(),
                                    std::prev(dtorBlock.end()));
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
                                            cir::GetGlobalOp getGlobalOp) {
  CIRBaseBuilderTy builder(getContext());

  std::optional<cir::ASTVarDeclInterface> astOption = globalOp.getAst();
  assert(astOption.has_value());
  cir::ASTVarDeclInterface varDecl = astOption.value();

  builder.setInsertionPointAfter(getGlobalOp);
  mlir::Block *getGlobalOpBlock = builder.getInsertionBlock();

  // Remove the terminator temporarily - we'll add it back at the end.
  mlir::Operation *ret = getGlobalOpBlock->getTerminator();
  ret->remove();
  builder.setInsertionPointAfter(getGlobalOp);

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

  // TLS variables need special handling - the guard must also be thread-local.
  if (varDecl.getTLSKind()) {
    globalOp->emitError("NYI: guarded initialization for thread-local statics");
    return;
  }

  // If we have a global variable with internal linkage and thread-safe statics
  // are disabled, we can just let the guard variable be of type i8.
  bool useInt8GuardVariable = !threadsafe && globalOp.hasInternalLinkage();
  if (useInt8GuardVariable) {
    globalOp->emitError("NYI: int8 guard variables for non-threadsafe statics");
    return;
  }

  // Guard variables are 64 bits in the generic ABI and size width on ARM
  // (i.e. 32-bit on AArch32, 64-bit on AArch64).
  if (useARMGuardVarABI()) {
    globalOp->emitError("NYI: ARM-style guard variables for static locals");
    return;
  }
  cir::IntType guardTy =
      cir::IntType::get(&getContext(), 64, /*isSigned=*/true);
  cir::CIRDataLayout dataLayout(mlirModule);
  clang::CharUnits guardAlignment =
      clang::CharUnits::fromQuantity(dataLayout.getABITypeAlign(guardTy));
  auto guardPtrTy = cir::PointerType::get(guardTy);

  // Create the guard variable if we don't already have it.
  cir::GlobalOp guard = getOrCreateStaticLocalDeclGuardAddress(
      builder, globalOp, varDecl, guardTy, guardAlignment);
  if (!guard) {
    // Error was already emitted, just restore the terminator and return.
    getGlobalOpBlock->push_back(ret);
    return;
  }

  mlir::Value guardPtr = builder.createGetGlobal(guard, /*threadLocal*/ false);

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
        getGlobalOp.getLoc(), bytePtr, guardAlignment.getAsAlign().value());

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
    if (useARMGuardVarABI()) {
      globalOp->emitError(
          "NYI: ARM-style guard variable check (bit 0 only) for static locals");
      return;
    }

    // Check if the first byte of the guard variable is zero.
    auto zero = builder.getConstantInt(
        getGlobalOp.getLoc(), mlir::cast<cir::IntType>(guardLoad.getType()), 0);
    auto needsInit = builder.createCompare(getGlobalOp.getLoc(),
                                           cir::CmpOpKind::eq, guardLoad, zero);

    // Build the guarded initialization inside an if block.
    cir::IfOp::create(builder, globalOp.getLoc(), needsInit,
                      /*withElseRegion=*/false,
                      [&](mlir::OpBuilder &, mlir::Location) {
                        emitCXXGuardedInitIf(builder, globalOp, varDecl,
                                             guardPtr, guardPtrTy, threadsafe);
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

void LoweringPreparePass::lowerGlobalOp(GlobalOp op) {
  // Static locals are handled separately via guard variables.
  if (op.getStaticLocalGuard())
    return;

  mlir::Region &ctorRegion = op.getCtorRegion();
  mlir::Region &dtorRegion = op.getDtorRegion();

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

  assert(!cir::MissingFeatures::opGlobalAnnotations());
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

  // Build name: __const.<func>.<var>
  std::string name = ("__const." + funcName + "." + varName).str();

  // Create the global constant.
  CIRBaseBuilderTy builder(getContext());

  // Use InsertionGuard to create the global at module level.
  builder.setInsertionPointToStart(mlirModule.getBody());

  // If a global with this name already exists (e.g. CIRGen materializes
  // constexpr locals as globals when their address is taken), reuse it.
  if (!mlir::SymbolTable::lookupSymbolIn(
          mlirModule, mlir::StringAttr::get(&getContext(), name))) {
    auto gv = cir::GlobalOp::create(
        builder, op.getLoc(), name, ty,
        /*isConstant=*/true,
        cir::LangAddressSpaceAttr::get(&getContext(),
                                       cir::LangAddressSpace::Default),
        cir::GlobalLinkageKind::PrivateLinkage);
    mlir::SymbolTable::setSymbolVisibility(
        gv, mlir::SymbolTable::Visibility::Private);
    gv.setInitialValueAttr(constant);
  }

  // Now replace the store with get_global + copy.
  builder.setInsertionPoint(op);

  auto ptrTy = cir::PointerType::get(ty);
  mlir::Value globalPtr =
      cir::GetGlobalOp::create(builder, op.getLoc(), ptrTy, name);

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
  } else if (auto getGlobal = mlir::dyn_cast<cir::GetGlobalOp>(op)) {
    // Handle static local variables with guard variables.
    // Only process GetGlobalOps inside function bodies, not in GlobalOp
    // regions.
    if (getGlobal.getStaticLocal() &&
        getGlobal->getParentOfType<cir::FuncOp>()) {
      auto globalOp = mlir::dyn_cast_or_null<cir::GlobalOp>(
          mlir::SymbolTable::lookupNearestSymbolFrom(getGlobal,
                                                     getGlobal.getNameAttr()));
      // Only process if the GlobalOp has static_local and the ctor region is
      // not empty. After handleStaticLocal processes a static local, the ctor
      // region is cleared. GetGlobalOps that were spliced from the ctor region
      // into the function will be skipped on subsequent iterations.
      if (globalOp && globalOp.getStaticLocalGuard() &&
          !globalOp.getCtorRegion().empty())
        handleStaticLocal(globalOp, getGlobal);
    }
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

  if (isHIP)
    assert(!cir::MissingFeatures::hipModuleCtor());
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
      fatbinType, builder.getStringAttr(gpuBinary->getBuffer())));
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
    llvm_unreachable("HIP Module Constructor Support");
  } else if (!astCtx->getLangOpts().GPURelocatableDeviceCode) {

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
    tmpString.setInitialValueAttr(ConstArrayAttr::get(
        strType, StringAttr::get(&getContext(), str + "\0")));
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

    if (isHIP) {
      llvm_unreachable("HIP kernel registration NYI");
    } else {
      mlir::Value hostFunc = builder.createBitcast(
          GetGlobalOp::create(
              builder, loc, PointerType::get(deviceStub.getFunctionType()),
              mlir::FlatSymbolRefAttr::get(deviceStub.getSymNameAttr())),
          voidPtrTy);
      builder.createCallOp(
          loc, cudaRegisterFunction,
          {fatbinHandle, hostFunc, deviceFunc, deviceFunc,
           ConstantOp::create(builder, loc, IntAttr::get(intTy, -1)),
           cirNullPtr, cirNullPtr, cirNullPtr, cirNullPtr, cirNullPtr});
    }
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
                  cir::MinusOp, cir::NotOp>(op))
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
