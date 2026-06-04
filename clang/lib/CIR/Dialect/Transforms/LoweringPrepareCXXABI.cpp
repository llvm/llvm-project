//====- LoweringPrepareCXXABI.cpp - Target ABI hooks for lowering prepare ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-ABI hooks declared in
// LoweringPrepareCXXABI.h.  The base implementation reports the requested
// lowering as not-yet-implemented; LoweringPrepareCXXABI::create returns a
// target-specific subclass when one is available.
//
//===----------------------------------------------------------------------===//

#include "LoweringPrepareCXXABI.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/TargetParser/Triple.h"

using namespace cir;

LoweringPrepareCXXABI::~LoweringPrepareCXXABI() = default;

mlir::LogicalResult LoweringPrepareCXXABI::lowerAggregateVAArg(
    CIRBaseBuilderTy &builder, cir::VAArgOp op,
    const cir::CIRDataLayout &datalayout) {
  op.emitError() << "ClangIR code gen Not Yet Implemented: "
                 << "va_arg of an aggregate type on this target";
  return mlir::failure();
}

namespace {

/// x86-64 System V implementation of the variadic aggregate hooks.
class LoweringPrepareX86_64CXXABI : public LoweringPrepareCXXABI {
public:
  mlir::LogicalResult
  lowerAggregateVAArg(CIRBaseBuilderTy &builder, cir::VAArgOp op,
                      const cir::CIRDataLayout &datalayout) override;
};

// Expand a `cir.va_arg` that produces an aggregate into the x86-64 System V
// register-save-area dance.  The generic lowering of `cir.va_arg` emits an
// `llvm.va_arg` instruction, which Selection-DAG cannot handle for aggregate
// types (it crashes with "Unknown type!").  Classic CodeGen open-codes this
// expansion at the AST level; CIR defers it to here so the LLVM lowering only
// ever sees a fully expanded form.
//
// Only aggregates whose eightbytes are all of INTEGER class (integers,
// pointers, bools) up to two eightbytes, plus larger integer-compatible
// aggregates that are always passed in memory (MEMORY class), are handled.
// Aggregates with floating-point members (which need SSE/`fp_offset`
// accounting) and over-aligned aggregates require eightbyte classification
// that is not implemented yet.  Rather than leave an `llvm.va_arg` that
// crashes the backend, those cases emit a not-yet-implemented diagnostic and
// return failure; see clang/docs/CIR/ABILowering.rst ("Variadic Aggregate
// Arguments").
mlir::LogicalResult LoweringPrepareX86_64CXXABI::lowerAggregateVAArg(
    CIRBaseBuilderTy &builder, cir::VAArgOp op,
    const cir::CIRDataLayout &datalayout) {
  auto recordTy = mlir::cast<cir::RecordType>(op.getType());

  auto reportNYI = [&](llvm::StringRef what) {
    op.emitError() << "ClangIR code gen Not Yet Implemented: " << what;
    return mlir::failure();
  };

  mlir::Type ty = op.getType();
  uint64_t size = datalayout.getTypeStoreSize(ty).getFixedValue();
  uint64_t tyAlign = datalayout.getABITypeAlign(ty).value();

  // Over-aligned aggregates need extra rounding that this path does not yet
  // implement.
  if (size == 0 || tyAlign > 8)
    return reportNYI("va_arg of an over-aligned aggregate type");

  // Determine how many general-purpose registers the aggregate would occupy.
  // A value of zero means the aggregate is always passed in memory.  Members
  // of floating-point type would land in the SSE register file and need
  // `fp_offset` accounting, which is not implemented yet.
  unsigned neededInt = 0;
  if (size <= 16) {
    for (mlir::Type member : recordTy.getMembers())
      if (!mlir::isa<cir::IntType, cir::PointerType, cir::BoolType>(member))
        return reportNYI(
            "va_arg of an aggregate type with non-integer members");
    neededInt = (size + 7) / 8;
  }

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value valist = op.getArgList();

  auto vaListRecTy = mlir::cast<cir::RecordType>(
      mlir::cast<cir::PointerType>(valist.getType()).getPointee());
  llvm::ArrayRef<mlir::Type> vaFields = vaListRecTy.getMembers();
  cir::IntType byteTy = builder.getSIntNTy(8);

  // Read the next argument from the overflow area and advance the cursor.
  // Returns an `i8*` to the argument slot.
  auto buildMemAddr = [&]() -> mlir::Value {
    mlir::Value overflowP = builder.createGetMember(
        loc, builder.getPointerTo(vaFields[2]), valist, "overflow_arg_area", 2);
    mlir::Value overflow = builder.createLoad(loc, overflowP);
    mlir::Value bytePtr = builder.createPtrBitcast(overflow, byteTy);
    uint64_t strideBytes = (size + 7) & ~UINT64_C(7);
    mlir::Value stride = builder.getSignedInt(loc, strideBytes, 32);
    mlir::Value next = builder.createPtrStride(loc, bytePtr, stride);
    builder.createStore(loc, next, overflowP);
    return bytePtr;
  };

  if (neededInt == 0) {
    mlir::Value addr = buildMemAddr();
    mlir::Value result =
        builder.createLoad(loc, builder.createPtrBitcast(addr, ty));
    op.replaceAllUsesWith(result);
    op.erase();
    return mlir::success();
  }

  mlir::OpBuilder::InsertPoint scopeIP;
  auto scopeOp = cir::ScopeOp::create(
      builder, loc,
      [&](mlir::OpBuilder &b, mlir::Type &yieldTy, mlir::Location l) {
        scopeIP = b.saveInsertionPoint();
        yieldTy = ty;
      });

  mlir::Block *contBlock = scopeIP.getBlock();
  mlir::Block *entryBlock = builder.createBlock(contBlock);
  mlir::Block *inRegBlock = builder.createBlock(contBlock);
  mlir::Block *inMemBlock = builder.createBlock(contBlock);

  // Decide whether enough general-purpose registers remain.  The register save
  // area holds the six GP registers in its first 48 bytes; `gp_offset` is the
  // byte offset of the next available one.
  builder.setInsertionPointToEnd(entryBlock);
  mlir::Value gpOffsetP = builder.createGetMember(
      loc, builder.getPointerTo(vaFields[0]), valist, "gp_offset", 0);
  mlir::Value gpOffset = builder.createLoad(loc, gpOffsetP);
  mlir::Value limit =
      builder.getConstantInt(loc, gpOffset.getType(), 48 - neededInt * 8);
  mlir::Value inRegs =
      builder.createCompare(loc, cir::CmpOpKind::le, gpOffset, limit);
  cir::BrCondOp::create(builder, loc, inRegs, inRegBlock, inMemBlock);

  // In registers: the slot lives at reg_save_area + gp_offset.  Bump gp_offset
  // past the registers this argument consumes.
  builder.setInsertionPointToEnd(inRegBlock);
  mlir::Value regSaveArea = builder.createLoad(
      loc, builder.createGetMember(loc, builder.getPointerTo(vaFields[3]),
                                   valist, "reg_save_area", 3));
  regSaveArea = builder.createPtrBitcast(regSaveArea, byteTy);
  mlir::Value regAddr = builder.createPtrStride(loc, regSaveArea, gpOffset);
  mlir::Value bump =
      builder.getConstantInt(loc, gpOffset.getType(), neededInt * 8);
  builder.createStore(loc, builder.createAdd(loc, gpOffset, bump), gpOffsetP);
  cir::BrOp::create(builder, loc, mlir::ValueRange{regAddr}, contBlock);

  // In memory: fetch from the overflow area.
  builder.setInsertionPointToEnd(inMemBlock);
  mlir::Value memAddr = buildMemAddr();
  cir::BrOp::create(builder, loc, mlir::ValueRange{memAddr}, contBlock);

  // Continuation: load the value from whichever slot was chosen.
  builder.setInsertionPointToStart(contBlock);
  mlir::Value resAddr = contBlock->addArgument(regAddr.getType(), loc);
  mlir::Value result =
      builder.createLoad(loc, builder.createPtrBitcast(resAddr, ty));
  cir::YieldOp::create(builder, loc, result);

  op.replaceAllUsesWith(scopeOp.getResult(0));
  op.erase();
  return mlir::success();
}

} // namespace

std::unique_ptr<LoweringPrepareCXXABI>
LoweringPrepareCXXABI::create(const llvm::Triple &triple) {
  if (triple.getArch() == llvm::Triple::x86_64)
    return std::make_unique<LoweringPrepareX86_64CXXABI>();

  // Targets without a specialized implementation fall back to the base, whose
  // hooks report not-yet-implemented.
  return std::make_unique<LoweringPrepareCXXABI>();
}
