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
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace cir;

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

namespace {
struct LoweringPreparePass : public LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  void runOnOp(Operation *op);
  void lowerGlobalOp(GlobalOp op);

  /// Build the function that initializes the specified global
  cir::FuncOp buildCXXGlobalVarDeclInitFunc(GlobalOp op);

  /// Build a module init function that calls all the dynamic initializers.
  void buildCXXGlobalInitFunc();

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

cir::FuncOp LoweringPreparePass::buildCXXGlobalVarDeclInitFunc(GlobalOp op) {
  SmallString<256> fnName;
  {
    std::unique_ptr<clang::MangleContext> MangleCtx(
        astCtx->createMangleContext());
    llvm::raw_svector_ostream Out(fnName);
    auto varDecl = op.getAst()->getAstDecl();
    MangleCtx->mangleDynamicInitializer(varDecl, Out);
    // Name numbering
    uint32_t cnt = dynamicInitializerNames[fnName]++;
    if (cnt)
      fnName += "." + llvm::Twine(cnt).str();
  }

  // Create a variable initialization function.
  mlir::OpBuilder builder(&getContext());
  builder.setInsertionPointAfter(op);
  auto fnType = mlir::cir::FuncType::get(
      {}, mlir::cir::VoidType::get(builder.getContext()));
  FuncOp f = builder.create<mlir::cir::FuncOp>(op.getLoc(), fnName, fnType);
  f.setLinkageAttr(mlir::cir::GlobalLinkageKindAttr::get(
      builder.getContext(), mlir::cir::GlobalLinkageKind::InternalLinkage));
  mlir::SymbolTable::setSymbolVisibility(
      f, mlir::SymbolTable::Visibility::Private);
  mlir::NamedAttrList attrs;
  f.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
      builder.getContext(), attrs.getDictionary(builder.getContext())));

  // move over the initialzation code of the ctor region.
  auto &block = op.getCtorRegion().front();
  mlir::Block *EntryBB = f.addEntryBlock();
  EntryBB->getOperations().splice(EntryBB->begin(), block.getOperations(),
                                  block.begin(), std::prev(block.end()));

  // Replace cir.yield with cir.return
  builder.setInsertionPointToEnd(EntryBB);
  auto &yieldOp = block.getOperations().back();
  assert(isa<YieldOp>(yieldOp));
  builder.create<ReturnOp>(yieldOp.getLoc());
  return f;
}

void LoweringPreparePass::lowerGlobalOp(GlobalOp op) {
  auto &ctorRegion = op.getCtorRegion();
  if (!ctorRegion.empty()) {
    // Build a variable initialization function and move the initialzation code
    // in the ctor region over.
    auto f = buildCXXGlobalVarDeclInitFunc(op);

    // Clear the ctor region
    ctorRegion.getBlocks().clear();

    // Add a function call to the variable initialization function.
    assert(!op.getAst()->getAstDecl()->getAttr<clang::InitPriorityAttr>() &&
           "custom initialization priority NYI");
    dynamicInitializers.push_back(f);
  }

  auto &dtorRegion = op.getDtorRegion();
  if (!dtorRegion.empty()) {
    // TODO: handle destructor
    // Clear the dtor region
    dtorRegion.getBlocks().clear();
  }
}

void LoweringPreparePass::buildCXXGlobalInitFunc() {
  if (dynamicInitializers.empty())
    return;

  SmallVector<mlir::Attribute, 4> attrs;
  for (auto &f : dynamicInitializers) {
    // TODO: handle globals with a user-specified initialzation priority.
    auto ctorAttr =
        mlir::cir::GlobalCtorAttr::get(&getContext(), f.getName());
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
      builder.create<mlir::cir::FuncOp>(theModule.getLoc(), fnName, fnType);
  f.setLinkageAttr(mlir::cir::GlobalLinkageKindAttr::get(
      builder.getContext(), mlir::cir::GlobalLinkageKind::ExternalLinkage));
  mlir::SymbolTable::setSymbolVisibility(
      f, mlir::SymbolTable::Visibility::Private);
  mlir::NamedAttrList extraAttrs;
  f.setExtraAttrsAttr(mlir::cir::ExtraFuncAttributesAttr::get(
      builder.getContext(), extraAttrs.getDictionary(builder.getContext())));

  builder.setInsertionPointToStart(f.addEntryBlock());
  for (auto &f : dynamicInitializers) {
    builder.create<mlir::cir::CallOp>(f.getLoc(), f);
  }

  builder.create<ReturnOp>(f.getLoc());
}

void LoweringPreparePass::runOnOp(Operation *op) {
  if (GlobalOp globalOp = cast<GlobalOp>(op)) {
    lowerGlobalOp(globalOp);
    return;
  }
}

void LoweringPreparePass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  auto* op = getOperation();
  if (isa<::mlir::ModuleOp>(op)) {
    theModule = cast<::mlir::ModuleOp>(op);
  }

  SmallVector<Operation *> opsToTransform;
  op->walk([&](Operation *op) {
    if (isa<GlobalOp>(op))
      opsToTransform.push_back(op);
  });

  for (auto *o : opsToTransform) {
    runOnOp(o);
  }

  buildCXXGlobalInitFunc();
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LoweringPreparePass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
