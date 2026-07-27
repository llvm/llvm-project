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

static void xformStdFindIntoMemchr(StdFindOp findOp,
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
      (enclosing && noBuiltinsForbid(enclosing, "memchr"))) {
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
  // Logical SPIR-V and the PS3 pair 64 bit pointers with a 32 bit size_t.
  // The GPU targets and BPF ship no C library that could provide the call.
  // TODO(cir): gate on the target size type and libcall availability.
  auto module = findOp->getParentOfType<mlir::ModuleOp>();
  auto tripleAttr = module ? module->getAttrOfType<mlir::StringAttr>(
                                 cir::CIRDialect::getTripleAttrName())
                           : nullptr;
  if (!tripleAttr)
    return;
  llvm::Triple triple(tripleAttr.getValue().str());
  if (!triple.isArch64Bit() || triple.isX32() || triple.isABIN32() ||
      triple.getEnvironment() == llvm::Triple::GNUILP32 || triple.isSPIRV() ||
      triple.isAMDGPU() || triple.isNVPTX() || triple.isBPF() ||
      triple.getOS() == llvm::Triple::Lv2) {
    return;
  }

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
      [&](StdFindOp findOp) { xformStdFindIntoMemchr(findOp, symbolTables); });
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}
