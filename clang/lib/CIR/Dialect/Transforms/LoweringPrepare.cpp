//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Attributes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetCXXABI.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/Path.h"

#include <memory>

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
  void lowerUnaryOp(cir::UnaryOp op);
  void lowerGlobalOp(cir::GlobalOp op);
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
    g.setGlobalVisibilityAttr(
        cir::VisibilityAttr::get(builder.getContext(), visibility));
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

  mlir::Value ac = builder.createBinop(loc, a, cir::BinOpKind::Mul, c); // a*c
  mlir::Value bd = builder.createBinop(loc, b, cir::BinOpKind::Mul, d); // b*d
  mlir::Value cc = builder.createBinop(loc, c, cir::BinOpKind::Mul, c); // c*c
  mlir::Value dd = builder.createBinop(loc, d, cir::BinOpKind::Mul, d); // d*d
  mlir::Value acbd =
      builder.createBinop(loc, ac, cir::BinOpKind::Add, bd); // ac+bd
  mlir::Value ccdd =
      builder.createBinop(loc, cc, cir::BinOpKind::Add, dd); // cc+dd
  mlir::Value resultReal =
      builder.createBinop(loc, acbd, cir::BinOpKind::Div, ccdd);

  mlir::Value bc = builder.createBinop(loc, b, cir::BinOpKind::Mul, c); // b*c
  mlir::Value ad = builder.createBinop(loc, a, cir::BinOpKind::Mul, d); // a*d
  mlir::Value bcad =
      builder.createBinop(loc, bc, cir::BinOpKind::Sub, ad); // bc-ad
  mlir::Value resultImag =
      builder.createBinop(loc, bcad, cir::BinOpKind::Div, ccdd);
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
    mlir::Value r = builder.createBinop(loc, d, cir::BinOpKind::Div,
                                        c); // r := d / c
    mlir::Value rd = builder.createBinop(loc, r, cir::BinOpKind::Mul, d); // r*d
    mlir::Value tmp = builder.createBinop(loc, c, cir::BinOpKind::Add,
                                          rd); // tmp := c + r*d

    mlir::Value br = builder.createBinop(loc, b, cir::BinOpKind::Mul, r); // b*r
    mlir::Value abr =
        builder.createBinop(loc, a, cir::BinOpKind::Add, br); // a + b*r
    mlir::Value e = builder.createBinop(loc, abr, cir::BinOpKind::Div, tmp);

    mlir::Value ar = builder.createBinop(loc, a, cir::BinOpKind::Mul, r); // a*r
    mlir::Value bar =
        builder.createBinop(loc, b, cir::BinOpKind::Sub, ar); // b - a*r
    mlir::Value f = builder.createBinop(loc, bar, cir::BinOpKind::Div, tmp);

    mlir::Value result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto falseBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    mlir::Value r = builder.createBinop(loc, c, cir::BinOpKind::Div,
                                        d); // r := c / d
    mlir::Value rc = builder.createBinop(loc, r, cir::BinOpKind::Mul, c); // r*c
    mlir::Value tmp = builder.createBinop(loc, d, cir::BinOpKind::Add,
                                          rc); // tmp := d + r*c

    mlir::Value ar = builder.createBinop(loc, a, cir::BinOpKind::Mul, r); // a*r
    mlir::Value arb =
        builder.createBinop(loc, ar, cir::BinOpKind::Add, b); // a*r + b
    mlir::Value e = builder.createBinop(loc, arb, cir::BinOpKind::Div, tmp);

    mlir::Value br = builder.createBinop(loc, b, cir::BinOpKind::Mul, r); // b*r
    mlir::Value bra =
        builder.createBinop(loc, br, cir::BinOpKind::Sub, a); // b*r - a
    mlir::Value f = builder.createBinop(loc, bra, cir::BinOpKind::Div, tmp);

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
  mlir::Value resultRealLhs =
      builder.createBinop(loc, lhsReal, cir::BinOpKind::Mul, rhsReal);
  mlir::Value resultRealRhs =
      builder.createBinop(loc, lhsImag, cir::BinOpKind::Mul, rhsImag);
  mlir::Value resultImagLhs =
      builder.createBinop(loc, lhsReal, cir::BinOpKind::Mul, rhsImag);
  mlir::Value resultImagRhs =
      builder.createBinop(loc, lhsImag, cir::BinOpKind::Mul, rhsReal);
  mlir::Value resultReal = builder.createBinop(
      loc, resultRealLhs, cir::BinOpKind::Sub, resultRealRhs);
  mlir::Value resultImag = builder.createBinop(
      loc, resultImagLhs, cir::BinOpKind::Add, resultImagRhs);
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

void LoweringPreparePass::lowerUnaryOp(cir::UnaryOp op) {
  mlir::Type ty = op.getType();
  if (!mlir::isa<cir::ComplexType>(ty))
    return;

  mlir::Location loc = op.getLoc();
  cir::UnaryOpKind opKind = op.getKind();

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  mlir::Value operand = op.getInput();
  mlir::Value operandReal = builder.createComplexReal(loc, operand);
  mlir::Value operandImag = builder.createComplexImag(loc, operand);

  mlir::Value resultReal;
  mlir::Value resultImag;

  switch (opKind) {
  case cir::UnaryOpKind::Inc:
  case cir::UnaryOpKind::Dec:
    resultReal = builder.createUnaryOp(loc, opKind, operandReal);
    resultImag = operandImag;
    break;

  case cir::UnaryOpKind::Plus:
  case cir::UnaryOpKind::Minus:
    resultReal = builder.createUnaryOp(loc, opKind, operandReal);
    resultImag = builder.createUnaryOp(loc, opKind, operandImag);
    break;

  case cir::UnaryOpKind::Not:
    resultReal = operandReal;
    resultImag =
        builder.createUnaryOp(loc, cir::UnaryOpKind::Minus, operandImag);
    break;
  }

  mlir::Value result = builder.createComplexCreate(loc, resultReal, resultImag);
  op.replaceAllUsesWith(result);
  op.erase();
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

static void lowerArrayDtorCtorIntoLoop(cir::CIRBaseBuilderTy &builder,
                                       clang::ASTContext *astCtx,
                                       mlir::Operation *op, mlir::Type eltTy,
                                       mlir::Value arrayAddr, uint64_t arrayLen,
                                       bool isCtor) {
  // Generate loop to call into ctor/dtor for every element.
  mlir::Location loc = op->getLoc();

  // TODO: instead of getting the size from the AST context, create alias for
  // PtrDiffTy and unify with CIRGen stuff.
  const unsigned sizeTypeSize =
      astCtx->getTypeSize(astCtx->getSignedSizeType());
  uint64_t endOffset = isCtor ? arrayLen : arrayLen - 1;
  mlir::Value endOffsetVal =
      builder.getUnsignedInt(loc, endOffset, sizeTypeSize);

  auto begin = cir::CastOp::create(builder, loc, eltTy,
                                   cir::CastKind::array_to_ptrdecay, arrayAddr);
  mlir::Value end =
      cir::PtrStrideOp::create(builder, loc, eltTy, begin, endOffsetVal);
  mlir::Value start = isCtor ? begin : end;
  mlir::Value stop = isCtor ? end : begin;

  mlir::Value tmpAddr = builder.createAlloca(
      loc, /*addr type*/ builder.getPointerTo(eltTy),
      /*var type*/ eltTy, "__array_idx", builder.getAlignmentAttr(1));
  builder.createStore(loc, start, tmpAddr);

  cir::DoWhileOp loop = builder.createDoWhile(
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

        cir::CallOp ctorCall;
        op->walk([&](cir::CallOp c) { ctorCall = c; });
        assert(ctorCall && "expected ctor call");

        // Array elements get constructed in order but destructed in reverse.
        mlir::Value stride;
        if (isCtor)
          stride = builder.getUnsignedInt(loc, 1, sizeTypeSize);
        else
          stride = builder.getSignedInt(loc, -1, sizeTypeSize);

        ctorCall->moveBefore(stride.getDefiningOp());
        ctorCall->setOperand(0, currentElement);
        auto nextElement = cir::PtrStrideOp::create(builder, loc, eltTy,
                                                    currentElement, stride);

        // Store the element pointer to the temporary variable
        builder.createStore(loc, nextElement, tmpAddr);
        builder.createYield(loc);
      });

  op->replaceAllUsesWith(loop);
  op->erase();
}

void LoweringPreparePass::lowerArrayDtor(cir::ArrayDtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  mlir::Type eltTy = op->getRegion(0).getArgument(0).getType();
  assert(!cir::MissingFeatures::vlas());
  auto arrayLen =
      mlir::cast<cir::ArrayType>(op.getAddr().getType().getPointee()).getSize();
  lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(), arrayLen,
                             false);
}

void LoweringPreparePass::lowerArrayCtor(cir::ArrayCtor op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  mlir::Type eltTy = op->getRegion(0).getArgument(0).getType();
  assert(!cir::MissingFeatures::vlas());
  auto arrayLen =
      mlir::cast<cir::ArrayType>(op.getAddr().getType().getPointee()).getSize();
  lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(), arrayLen,
                             true);
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
    auto gv = cir::GlobalOp::create(builder, op.getLoc(), name, ty,
                                    /*isConstant=*/true,
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
  } else if (auto unary = mlir::dyn_cast<cir::UnaryOp>(op)) {
    lowerUnaryOp(unary);
  } else if (auto callOp = dyn_cast<cir::CallOp>(op)) {
    lowerTrivialCopyCall(callOp);
  } else if (auto storeOp = dyn_cast<cir::StoreOp>(op)) {
    lowerStoreOfConstAggregate(storeOp);
  } else if (auto fnOp = dyn_cast<cir::FuncOp>(op)) {
    if (auto globalCtor = fnOp.getGlobalCtorPriority())
      globalCtorList.emplace_back(fnOp.getName(), globalCtor.value());
    else if (auto globalDtor = fnOp.getGlobalDtorPriority())
      globalDtorList.emplace_back(fnOp.getName(), globalDtor.value());
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
                  cir::StoreOp, cir::UnaryOp>(op))
      opsToTransform.push_back(op);
  });

  for (mlir::Operation *o : opsToTransform)
    runOnOp(o);

  buildCXXGlobalInitFunc();
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
