//===- LibOpt.cpp - Optimize CIR raised C/C++ library idioms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes C/C++ standard library idioms in Clang IR.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LIBOPT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct LibOptPass : public impl::LibOptBase<LibOptPass> {
  LibOptPass() = default;
  mlir::LogicalResult
  initializeOptions(llvm::StringRef options,
                    llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                        errorHandler) override;
  void runOnOperation() override;

  // Raw libopt option string forwarded by the frontend. This will later control
  // which optimizations the pass enables.
  std::string optimizationOptions;
};
} // namespace

mlir::LogicalResult LibOptPass::initializeOptions(
    llvm::StringRef options,
    llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)>) {
  optimizationOptions = options.str();
  // TODO(cir): Parse options to select the active transformations for the
  // pass.
  return mlir::success();
}

static void rewriteStdFindToMemchr(StdFindOp findOp,
                                   mlir::SymbolTableCollection &symbolTables) {
  auto iterTy = mlir::dyn_cast<cir::PointerType>(findOp.getResult().getType());
  if (!iterTy || iterTy.getAddrSpace())
    return;
  auto elemTy = mlir::dyn_cast<cir::IntType>(iterTy.getPointee());
  if (!elemTy || elemTy.getWidth() != 8)
    return;

  auto patternPtrTy =
      mlir::dyn_cast<cir::PointerType>(findOp.getPattern().getType());
  if (!patternPtrTy || patternPtrTy.getPointee() != elemTy)
    return;

  // LibOpt runs before LoweringPrepare, so a global initializer is still a
  // cir.global here. Anything else is not a shape CIRGen produces.
  auto enclosing = findOp->getParentOfType<cir::FuncOp>();
  auto enclosingGlobal = findOp->getParentOfType<cir::GlobalOp>();
  if (!enclosing && !enclosingGlobal)
    return;

  // No builtin state rides on the raised call and on the enclosing function.
  // A global initializer has no function to carry the list.
  if (isNoBuiltin(findOp, "memchr") ||
      (enclosing && noBuiltinListDisables(enclosing, "memchr")))
    return;

  // An enum or atomic element also lowers to a byte wide integer.
  auto callee = symbolTables.lookupNearestSymbolFrom<cir::FuncOp>(
      findOp, findOp.getOriginalFnAttr());
  auto funcIdentity = mlir::dyn_cast_if_present<cir::FuncIdentityAttr>(
      callee ? callee.getFuncInfoAttr() : mlir::Attribute());
  if (!funcIdentity || funcIdentity.getKind() != cir::KnownFuncKind::StdFind ||
      !funcIdentity.getNarrowCharParams()) {
    return;
  }

  auto moduleOp = findOp->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return;

  auto tripleAttr = moduleOp->getAttrOfType<mlir::StringAttr>(
      cir::CIRDialect::getTripleAttrName());
  if (!tripleAttr)
    return;

  llvm::Triple triple(tripleAttr.getValue().str());

  // cir.libc.memchr currently requires a 64 bit length. Use the AST size_t
  // width recorded by CIRGen instead of inferring it from target properties.
  auto sizeWidthAttr = moduleOp->getAttrOfType<mlir::IntegerAttr>(
      cir::CIRDialect::getSizeTypeWidthAttrName());
  bool sizeTypeMismatch = !sizeWidthAttr ||
                          !sizeWidthAttr.getType().isSignlessInteger(32) ||
                          sizeWidthAttr.getInt() != 64;

  // The current memchr lowering supplies no target ABI extension attributes, so
  // restrict the rewrite to targets known not to need them.
  // TODO(cir): use target-aware ABI and libcall availability information.
  bool abiSafeTarget =
      triple.getArch() == llvm::Triple::x86_64 || triple.isAArch64();

  // AArch64 GNUILP32 AST types disagree with LLVM's data layout.
  bool unsupportedABI =
      triple.isAArch64() && triple.getEnvironment() == llvm::Triple::GNUILP32;

  if (sizeTypeMismatch || !abiSafeTarget || unsupportedABI)
    return;

  // Lowering resolves memchr in the module symbol table. An existing symbol may
  // collide with the introduced libcall or carry incompatible semantics.
  if (symbolTables.lookupSymbolIn(
          moduleOp, mlir::StringAttr::get(findOp.getContext(), "memchr")))
    return;

  mlir::Location loc = findOp.getLoc();
  mlir::Value first = findOp.getFirst();
  mlir::Value last = findOp.getLast();
  CIRBaseBuilderTy builder(*findOp.getContext());
  builder.setInsertionPointAfter(findOp);

  // C requires the pointer argument to be valid even when the length is zero,
  // and an empty range is allowed to be a pair of null pointers.
  mlir::Value isEmpty =
      builder.createCompare(loc, cir::CmpOpKind::eq, first, last);
  mlir::Value result =
      cir::TernaryOp::create(
          builder, loc, isEmpty,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, last);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value src =
                builder.createBitcast(loc, first, builder.getVoidPtrTy());
            mlir::Value pattern = builder.createIntCast(
                builder.createLoad(loc, findOp.getPattern()),
                builder.getSIntNTy(32));
            mlir::Value len = cir::PtrDiffOp::create(
                builder, loc, builder.getUIntNTy(64), last, first);
            mlir::Value res =
                cir::MemChrOp::create(builder, loc, src, pattern, len);
            res = builder.createBitcast(loc, res, iterTy);
            builder.createYield(
                loc, builder.createSelect(loc, builder.createPtrIsNull(res),
                                          last, res));
          })
          .getResult();
  findOp.getResult().replaceAllUsesWith(result);
  findOp.erase();
}

void LibOptPass::runOnOperation() {
  mlir::SymbolTableCollection symbolTables;
  getOperation()->walk(
      [&](StdFindOp findOp) { rewriteStdFindToMemchr(findOp, symbolTables); });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
