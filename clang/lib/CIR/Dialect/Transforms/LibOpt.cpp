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

  // No builtin state rides on both the raised call and the enclosing function.
  auto enclosing = findOp->getParentOfType<cir::FuncOp>();
  if (isNoBuiltin(findOp, "memchr") ||
      (enclosing && noBuiltinListDisables(enclosing, "memchr"))) {
    return;
  }

  // An enum or atomic element also lowers to a byte wide integer.
  auto callee = symbolTables.lookupNearestSymbolFrom<cir::FuncOp>(
      findOp, findOp.getOriginalFnAttr());
  auto funcIdentity = mlir::dyn_cast_if_present<cir::FuncIdentityAttr>(
      callee ? callee.getFuncInfoAttr() : mlir::Attribute());
  if (!funcIdentity || funcIdentity.getKind() != cir::KnownFuncKind::StdFind ||
      !funcIdentity.getNarrowCharParams()) {
    return;
  }

  // cir.libc.memchr fixes the value and length widths at 32 and 64 bits.
  // TODO(cir): gate on the target size type and libcall availability.
  auto moduleOp = findOp->getParentOfType<mlir::ModuleOp>();
  auto tripleAttr = moduleOp ? moduleOp->getAttrOfType<mlir::StringAttr>(
                                   cir::CIRDialect::getTripleAttrName())
                             : nullptr;
  if (!tripleAttr)
    return;
  llvm::Triple triple(tripleAttr.getValue().str());

  // size_t is not 64 bits on the 32 bit archs, the ILP32 on 64 ABIs, and
  // the PS3, so the length argument would have the wrong width there.
  bool sizeTypeMismatch = !triple.isArch64Bit() || triple.isX32() ||
                          triple.isABIN32() ||
                          triple.getEnvironment() == llvm::Triple::GNUILP32 ||
                          triple.getOS() == llvm::Triple::Lv2;

  // These targets ship no C library that could provide the new call.
  bool noCLibrary = triple.isGPU() || triple.isBPF();

  if (sizeTypeMismatch || noCLibrary)
    return;

  mlir::Location loc = findOp.getLoc();
  mlir::Value first = findOp.getFirst();
  mlir::Value last = findOp.getLast();
  CIRBaseBuilderTy builder(*findOp.getContext());
  builder.setInsertionPointAfter(findOp.getOperation());

  // void *memchr(const void *s, int c, size_t n)
  mlir::Value src = builder.createBitcast(loc, first, builder.getVoidPtrTy());
  mlir::Value pattern = builder.createIntCast(
      builder.createLoad(loc, findOp.getPattern()), builder.getSIntNTy(32));
  mlir::Value len =
      cir::PtrDiffOp::create(builder, loc, builder.getUIntNTy(64), last, first);
  mlir::Value res = cir::MemChrOp::create(builder, loc, src, pattern, len);
  res = builder.createBitcast(loc, res, iterTy);

  mlir::Value result =
      builder.createSelect(loc, builder.createPtrIsNull(res), last, res);
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
