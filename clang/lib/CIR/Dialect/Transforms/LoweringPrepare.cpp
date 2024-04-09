//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace mlir::cir;

static SmallString<128> getTransformedFileName(ModuleOp theModule) {
  SmallString<128> FileName;

  if (theModule.getSymName()) {
    FileName = llvm::sys::path::filename(theModule.getSymName()->str());
  }

  if (FileName.empty())
    FileName = "<null>";

  for (size_t i = 0; i < FileName.size(); ++i) {
    // Replace everything that's not [a-zA-Z0-9._] with a _. This set happens
    // to be the set of C preprocessing numbers.
    if (!clang::isPreprocessingNumberBody(FileName[i]))
      FileName[i] = '_';
  }

  return FileName;
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

namespace {

struct LoweringPreparePass : public LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  void runOnOp(Operation *op);
  void lowerThreeWayCmpOp(CmpThreeWayOp op);
  void lowerGlobalOp(GlobalOp op);
  void lowerStdFindOp(StdFindOp op);
  void lowerIterBeginOp(IterBeginOp op);
  void lowerIterEndOp(IterEndOp op);
  void lowerArrayDtor(ArrayDtor op);
  void lowerArrayCtor(ArrayCtor op);

  /// Build the function that initializes the specified global
  FuncOp buildCXXGlobalVarDeclInitFunc(GlobalOp op);

  /// Build a module init function that calls all the dynamic initializers.
  void buildCXXGlobalInitFunc();

  FuncOp
  buildRuntimeFunction(mlir::OpBuilder &builder, llvm::StringRef name,
                       mlir::Location loc, mlir::cir::FuncType type,
                       mlir::cir::GlobalLinkageKind linkage =
                           mlir::cir::GlobalLinkageKind::ExternalLinkage);

  GlobalOp
  buildRuntimeVariable(mlir::OpBuilder &Builder, llvm::StringRef Name,
                       mlir::Location Loc, mlir::Type type,
                       mlir::cir::GlobalLinkageKind Linkage =
                           mlir::cir::GlobalLinkageKind::ExternalLinkage);

  ///
  /// AST related
  /// -----------

  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;

  /// Tracks existing dynamic initializers.
  llvm::StringMap<uint32_t> dynamicInitializerNames;
  llvm::SmallVector<FuncOp, 4> dynamicInitializers;
};
} // namespace

GlobalOp LoweringPreparePass::buildRuntimeVariable(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    mlir::Type type, mlir::cir::GlobalLinkageKind linkage) {
  GlobalOp g = dyn_cast_or_null<GlobalOp>(SymbolTable::lookupNearestSymbolFrom(
      theModule, StringAttr::get(theModule->getContext(), name)));
  if (!g) {
    g = builder.create<mlir::cir::GlobalOp>(loc, name, type);
    g.setLinkageAttr(
        mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

FuncOp LoweringPreparePass::buildRuntimeFunction(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    mlir::cir::FuncType type, mlir::cir::GlobalLinkageKind linkage) {
  FuncOp f = dyn_cast_or_null<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      theModule, StringAttr::get(theModule->getContext(), name)));
  if (!f) {
    f = builder.create<mlir::cir::FuncOp>(loc, name, type);
    f.setLinkageAttr(
        mlir::cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);
    mlir::NamedAttrList attrs;
    f.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
        builder.getContext(), attrs.getDictionary(builder.getContext())));
  }
  return f;
}

FuncOp LoweringPreparePass::buildCXXGlobalVarDeclInitFunc(GlobalOp op) {
  SmallString<256> fnName;
  {
    llvm::raw_svector_ostream Out(fnName);
    op.getAst()->mangleDynamicInitializer(Out);
    // Name numbering
    uint32_t cnt = dynamicInitializerNames[fnName]++;
    if (cnt)
      fnName += "." + llvm::Twine(cnt).str();
  }

  // Create a variable initialization function.
  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointAfter(op);
  auto voidTy = ::mlir::cir::VoidType::get(builder.getContext());
  auto fnType = mlir::cir::FuncType::get({}, voidTy);
  FuncOp f =
      buildRuntimeFunction(builder, fnName, op.getLoc(), fnType,
                           mlir::cir::GlobalLinkageKind::InternalLinkage);

  // Move over the initialzation code of the ctor region.
  auto &block = op.getCtorRegion().front();
  mlir::Block *entryBB = f.addEntryBlock();
  entryBB->getOperations().splice(entryBB->begin(), block.getOperations(),
                                  block.begin(), std::prev(block.end()));

  // Register the destructor call with __cxa_atexit
  auto &dtorRegion = op.getDtorRegion();
  if (!dtorRegion.empty()) {
    assert(op.getAst() &&
           op.getAst()->getTLSKind() == clang::VarDecl::TLS_None && " TLS NYI");
    // Create a variable that binds the atexit to this shared object.
    builder.setInsertionPointToStart(&theModule.getBodyRegion().front());
    auto Handle = buildRuntimeVariable(builder, "__dso_handle", op.getLoc(),
                                       builder.getI8Type());

    // Look for the destructor call in dtorBlock
    auto &dtorBlock = dtorRegion.front();
    mlir::cir::CallOp dtorCall;
    for (auto op : reverse(dtorBlock.getOps<mlir::cir::CallOp>())) {
      dtorCall = op;
      break;
    }
    assert(dtorCall && "Expected a dtor call");
    FuncOp dtorFunc = getCalledFunction(dtorCall);
    assert(dtorFunc &&
           mlir::isa<ASTCXXDestructorDeclInterface>(*dtorFunc.getAst()) &&
           "Expected a dtor call");

    // Create a runtime helper function:
    //    extern "C" int __cxa_atexit(void (*f)(void *), void *p, void *d);
    auto voidPtrTy =
        ::mlir::cir::PointerType::get(builder.getContext(), voidTy);
    auto voidFnTy = mlir::cir::FuncType::get({voidPtrTy}, voidTy);
    auto voidFnPtrTy =
        ::mlir::cir::PointerType::get(builder.getContext(), voidFnTy);
    auto HandlePtrTy =
        mlir::cir::PointerType::get(builder.getContext(), Handle.getSymType());
    auto fnAtExitType = mlir::cir::FuncType::get(
        {voidFnPtrTy, voidPtrTy, HandlePtrTy},
        mlir::cir::VoidType::get(builder.getContext()));
    const char *nameAtExit = "__cxa_atexit";
    FuncOp fnAtExit =
        buildRuntimeFunction(builder, nameAtExit, op.getLoc(), fnAtExitType);

    // Replace the dtor call with a call to __cxa_atexit(&dtor, &var,
    // &__dso_handle)
    builder.setInsertionPointAfter(dtorCall);
    mlir::Value args[3];
    auto dtorPtrTy = mlir::cir::PointerType::get(builder.getContext(),
                                                 dtorFunc.getFunctionType());
    // dtorPtrTy
    args[0] = builder.create<mlir::cir::GetGlobalOp>(
        dtorCall.getLoc(), dtorPtrTy, dtorFunc.getSymName());
    args[0] = builder.create<mlir::cir::CastOp>(
        dtorCall.getLoc(), voidFnPtrTy, mlir::cir::CastKind::bitcast, args[0]);
    args[1] = builder.create<mlir::cir::CastOp>(dtorCall.getLoc(), voidPtrTy,
                                                mlir::cir::CastKind::bitcast,
                                                dtorCall.getArgOperand(0));
    args[2] = builder.create<mlir::cir::GetGlobalOp>(
        Handle.getLoc(), HandlePtrTy, Handle.getSymName());
    builder.create<mlir::cir::CallOp>(dtorCall.getLoc(), fnAtExit, args);
    dtorCall->erase();
    entryBB->getOperations().splice(entryBB->end(), dtorBlock.getOperations(),
                                    dtorBlock.begin(),
                                    std::prev(dtorBlock.end()));
  }

  // Replace cir.yield with cir.return
  builder.setInsertionPointToEnd(entryBB);
  auto &yieldOp = block.getOperations().back();
  assert(isa<YieldOp>(yieldOp));
  builder.create<ReturnOp>(yieldOp.getLoc());
  return f;
}

void LoweringPreparePass::lowerThreeWayCmpOp(CmpThreeWayOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  auto loc = op->getLoc();
  auto cmpInfo = op.getInfo();

  auto buildCmpRes = [&](int64_t value) -> mlir::Value {
    return builder.create<mlir::cir::ConstantOp>(
        loc, op.getType(), mlir::cir::IntAttr::get(op.getType(), value));
  };
  auto ltRes = buildCmpRes(cmpInfo.getLt());
  auto eqRes = buildCmpRes(cmpInfo.getEq());
  auto gtRes = buildCmpRes(cmpInfo.getGt());

  auto buildCmp = [&](CmpOpKind kind) -> mlir::Value {
    auto ty = BoolType::get(&getContext());
    return builder.create<mlir::cir::CmpOp>(loc, ty, kind, op.getLhs(),
                                            op.getRhs());
  };
  auto buildSelect = [&](mlir::Value condition, mlir::Value trueResult,
                         mlir::Value falseResult) -> mlir::Value {
    return builder
        .create<mlir::cir::TernaryOp>(
            loc, condition,
            [&](OpBuilder &, Location) {
              builder.create<mlir::cir::YieldOp>(loc, trueResult);
            },
            [&](OpBuilder &, Location) {
              builder.create<mlir::cir::YieldOp>(loc, falseResult);
            })
        .getResult();
  };

  mlir::Value transformedResult;
  if (cmpInfo.getOrdering() == CmpOrdering::Strong) {
    // Strong ordering.
    auto lt = buildCmp(CmpOpKind::lt);
    auto eq = buildCmp(CmpOpKind::eq);
    auto selectOnEq = buildSelect(eq, eqRes, gtRes);
    transformedResult = buildSelect(lt, ltRes, selectOnEq);
  } else {
    // Partial ordering.
    auto unorderedRes = buildCmpRes(cmpInfo.getUnordered().value());

    auto lt = buildCmp(CmpOpKind::lt);
    auto eq = buildCmp(CmpOpKind::eq);
    auto gt = buildCmp(CmpOpKind::gt);
    auto selectOnEq = buildSelect(eq, eqRes, unorderedRes);
    auto selectOnGt = buildSelect(gt, gtRes, selectOnEq);
    transformedResult = buildSelect(lt, ltRes, selectOnGt);
  }

  op.replaceAllUsesWith(transformedResult);
  op.erase();
}

void LoweringPreparePass::lowerGlobalOp(GlobalOp op) {
  auto &ctorRegion = op.getCtorRegion();
  auto &dtorRegion = op.getDtorRegion();

  if (!ctorRegion.empty() || !dtorRegion.empty()) {
    // Build a variable initialization function and move the initialzation code
    // in the ctor region over.
    auto f = buildCXXGlobalVarDeclInitFunc(op);

    // Clear the ctor and dtor region
    ctorRegion.getBlocks().clear();
    dtorRegion.getBlocks().clear();

    // Add a function call to the variable initialization function.
    assert(!hasAttr<clang::InitPriorityAttr>(
               mlir::cast<ASTDeclInterface>(*op.getAst())) &&
           "custom initialization priority NYI");
    dynamicInitializers.push_back(f);
  }
}

void LoweringPreparePass::buildCXXGlobalInitFunc() {
  if (dynamicInitializers.empty())
    return;

  SmallVector<mlir::Attribute, 4> attrs;
  for (auto &f : dynamicInitializers) {
    // TODO: handle globals with a user-specified initialzation priority.
    auto ctorAttr = mlir::cir::GlobalCtorAttr::get(&getContext(), f.getName());
    attrs.push_back(ctorAttr);
  }

  theModule->setAttr("cir.globalCtors",
                     mlir::ArrayAttr::get(&getContext(), attrs));

  SmallString<256> fnName;
  // Include the filename in the symbol name. Including "sub_" matches gcc
  // and makes sure these symbols appear lexicographically behind the symbols
  // with priority emitted above.  Module implementation units behave the same
  // way as a non-modular TU with imports.
  // TODO: check CXX20ModuleInits
  if (astCtx->getCurrentNamedModule() &&
      !astCtx->getCurrentNamedModule()->isModuleImplementation()) {
    llvm::raw_svector_ostream Out(fnName);
    std::unique_ptr<clang::MangleContext> MangleCtx(
        astCtx->createMangleContext());
    cast<clang::ItaniumMangleContext>(*MangleCtx)
        .mangleModuleInitializer(astCtx->getCurrentNamedModule(), Out);
  } else {
    fnName += "_GLOBAL__sub_I_";
    fnName += getTransformedFileName(theModule);
  }

  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointToEnd(&theModule.getBodyRegion().back());
  auto fnType = mlir::cir::FuncType::get(
      {}, mlir::cir::VoidType::get(builder.getContext()));
  FuncOp f =
      buildRuntimeFunction(builder, fnName, theModule.getLoc(), fnType,
                           mlir::cir::GlobalLinkageKind::ExternalLinkage);
  builder.setInsertionPointToStart(f.addEntryBlock());
  for (auto &f : dynamicInitializers) {
    builder.create<mlir::cir::CallOp>(f.getLoc(), f);
  }

  builder.create<ReturnOp>(f.getLoc());
}

static void lowerArrayDtorCtorIntoLoop(CIRBaseBuilderTy &builder,
                                       mlir::Operation *op, mlir::Type eltTy,
                                       mlir::Value arrayAddr,
                                       uint64_t arrayLen) {
  // Generate loop to call into ctor/dtor for every element.
  auto loc = op->getLoc();

  // TODO: instead of fixed integer size, create alias for PtrDiffTy and unify
  // with CIRGen stuff.
  auto ptrDiffTy =
      mlir::cir::IntType::get(builder.getContext(), 64, /*signed=*/false);
  auto numArrayElementsConst = builder.create<mlir::cir::ConstantOp>(
      loc, ptrDiffTy, mlir::cir::IntAttr::get(ptrDiffTy, arrayLen));

  auto begin = builder.create<mlir::cir::CastOp>(
      loc, eltTy, mlir::cir::CastKind::array_to_ptrdecay, arrayAddr);
  mlir::Value end = builder.create<mlir::cir::PtrStrideOp>(
      loc, eltTy, begin, numArrayElementsConst);

  auto tmpAddr = builder.create<mlir::cir::AllocaOp>(
      loc, /*addr type*/ builder.getPointerTo(eltTy),
      /*var type*/ eltTy, "__array_idx",
      builder.getSizeFromCharUnits(builder.getContext(),
                                   clang::CharUnits::One()),
      nullptr);
  builder.create<mlir::cir::StoreOp>(loc, begin, tmpAddr);

  auto loop = builder.createDoWhile(
      loc,
      /*condBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto currentElement =
            b.create<mlir::cir::LoadOp>(loc, eltTy, tmpAddr.getResult());
        mlir::Type boolTy = mlir::cir::BoolType::get(b.getContext());
        auto cmp = builder.create<mlir::cir::CmpOp>(
            loc, boolTy, mlir::cir::CmpOpKind::eq, currentElement, end);
        builder.createCondition(cmp);
      },
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto currentElement =
            b.create<mlir::cir::LoadOp>(loc, eltTy, tmpAddr.getResult());

        CallOp ctorCall;
        op->walk([&](CallOp c) { ctorCall = c; });
        assert(ctorCall && "expected ctor call");

        auto one = builder.create<mlir::cir::ConstantOp>(
            loc, ptrDiffTy, mlir::cir::IntAttr::get(ptrDiffTy, 1));

        ctorCall->moveAfter(one);
        ctorCall->setOperand(0, currentElement);

        // Advance pointer and store them to temporary variable
        auto nextElement = builder.create<mlir::cir::PtrStrideOp>(
            loc, eltTy, currentElement, one);
        b.create<mlir::cir::StoreOp>(loc, nextElement, tmpAddr);
        builder.createYield(loc);
      });

  op->replaceAllUsesWith(loop);
  op->erase();
}

void LoweringPreparePass::lowerArrayDtor(ArrayDtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  auto eltTy = op->getRegion(0).getArgument(0).getType();
  auto arrayLen = op.getAddr()
                      .getType()
                      .cast<mlir::cir::PointerType>()
                      .getPointee()
                      .cast<mlir::cir::ArrayType>()
                      .getSize();
  lowerArrayDtorCtorIntoLoop(builder, op, eltTy, op.getAddr(), arrayLen);
}

void LoweringPreparePass::lowerArrayCtor(ArrayCtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  auto eltTy = op->getRegion(0).getArgument(0).getType();
  auto arrayLen = op.getAddr()
                      .getType()
                      .cast<mlir::cir::PointerType>()
                      .getPointee()
                      .cast<mlir::cir::ArrayType>()
                      .getSize();
  lowerArrayDtorCtorIntoLoop(builder, op, eltTy, op.getAddr(), arrayLen);
}

void LoweringPreparePass::lowerStdFindOp(StdFindOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());
  auto call = builder.create<mlir::cir::CallOp>(
      op.getLoc(), op.getOriginalFnAttr(), op.getResult().getType(),
      mlir::ValueRange{op.getOperand(0), op.getOperand(1), op.getOperand(2)});

  op.replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::lowerIterBeginOp(IterBeginOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());
  auto call = builder.create<mlir::cir::CallOp>(
      op.getLoc(), op.getOriginalFnAttr(), op.getResult().getType(),
      mlir::ValueRange{op.getOperand()});

  op.replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::lowerIterEndOp(IterEndOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());
  auto call = builder.create<mlir::cir::CallOp>(
      op.getLoc(), op.getOriginalFnAttr(), op.getResult().getType(),
      mlir::ValueRange{op.getOperand()});

  op.replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::runOnOp(Operation *op) {
  if (auto threeWayCmp = dyn_cast<CmpThreeWayOp>(op)) {
    lowerThreeWayCmpOp(threeWayCmp);
  } else if (auto getGlobal = dyn_cast<GlobalOp>(op)) {
    lowerGlobalOp(getGlobal);
  } else if (auto stdFind = dyn_cast<StdFindOp>(op)) {
    lowerStdFindOp(stdFind);
  } else if (auto iterBegin = dyn_cast<IterBeginOp>(op)) {
    lowerIterBeginOp(iterBegin);
  } else if (auto iterEnd = dyn_cast<IterEndOp>(op)) {
    lowerIterEndOp(iterEnd);
  } else if (auto arrayCtor = dyn_cast<ArrayCtor>(op)) {
    lowerArrayCtor(arrayCtor);
  } else if (auto arrayDtor = dyn_cast<ArrayDtor>(op)) {
    lowerArrayDtor(arrayDtor);
  }
}

void LoweringPreparePass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op)) {
    theModule = cast<::mlir::ModuleOp>(op);
  }

  SmallVector<Operation *> opsToTransform;
  op->walk([&](Operation *op) {
    if (isa<CmpThreeWayOp, GlobalOp, StdFindOp, IterBeginOp, IterEndOp,
            ArrayCtor, ArrayDtor>(op))
      opsToTransform.push_back(op);
  });

  for (auto *o : opsToTransform)
    runOnOp(o);

  buildCXXGlobalInitFunc();
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
