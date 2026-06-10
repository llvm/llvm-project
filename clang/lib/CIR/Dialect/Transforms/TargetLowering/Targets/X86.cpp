//===- X86.cpp - Emit CIR for x86-64 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TargetLoweringInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include <cstdint>

namespace cir {

namespace {

/// x86-64 System V target hooks.
class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  mlir::Value
  lowerAggregateVAArg(CIRBaseBuilderTy &builder, cir::VAArgOp op,
                      mlir::Value valist,
                      const cir::CIRDataLayout &dataLayout) const override;
};

// Expand a `cir.va_arg` that produces an aggregate into the x86-64 System V
// register-save-area dance.  The generic lowering of `cir.va_arg` emits an
// `llvm.va_arg` instruction, which Selection-DAG cannot handle for aggregate
// types (it crashes with "Unknown type!").  Classic CodeGen open-codes this
// expansion at the AST level; CIR defers it to the CXXABILowering pass so the
// LLVM lowering only ever sees a fully expanded form.
//
// Only aggregates whose eightbytes are all of INTEGER class (integers,
// pointers, bools) up to two eightbytes, plus larger integer-compatible
// aggregates that are always passed in memory (MEMORY class), are handled.
// Aggregates with floating-point members (which need SSE/`fp_offset`
// accounting) and over-aligned aggregates require eightbyte classification
// that is not implemented yet.  Rather than leave an `llvm.va_arg` that
// crashes the backend, those cases emit a not-yet-implemented diagnostic and
// return a null value; see clang/docs/CIR/ABILowering.rst ("Variadic
// Aggregate Arguments").
mlir::Value X86_64TargetLoweringInfo::lowerAggregateVAArg(
    CIRBaseBuilderTy &builder, cir::VAArgOp op, mlir::Value valist,
    const cir::CIRDataLayout &dataLayout) const {
  auto recordTy = mlir::cast<cir::RecordType>(op.getType());

  auto reportNYI = [&](llvm::StringRef what) -> mlir::Value {
    op.emitError() << "ClangIR code gen Not Yet Implemented: " << what;
    return {};
  };

  mlir::Type ty = op.getType();
  uint64_t size = dataLayout.getTypeStoreSize(ty).getFixedValue();
  uint64_t tyAlign = dataLayout.getABITypeAlign(ty).value();

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

  auto vaListRecTy = mlir::cast<cir::RecordType>(
      mlir::cast<cir::PointerType>(valist.getType()).getPointee());
  llvm::ArrayRef<mlir::Type> vaFields = vaListRecTy.getMembers();
  cir::IntType byteTy = builder.getSIntNTy(8);

  // Read the next argument from the overflow area and advance the cursor.
  // Returns an `i8*` to the argument slot.
  auto buildMemAddr = [&](CIRBaseBuilderTy &b) -> mlir::Value {
    mlir::Value overflowP = b.createGetMember(loc, b.getPointerTo(vaFields[2]),
                                              valist, "overflow_arg_area", 2);
    mlir::Value overflow = b.createLoad(loc, overflowP);
    mlir::Value bytePtr = b.createPtrBitcast(overflow, byteTy);
    uint64_t strideBytes = (size + 7) & ~UINT64_C(7);
    mlir::Value stride = b.getSignedInt(loc, strideBytes, 32);
    mlir::Value next = b.createPtrStride(loc, bytePtr, stride);
    b.createStore(loc, next, overflowP);
    return bytePtr;
  };

  // Always passed in memory: read straight from the overflow area.
  if (neededInt == 0) {
    mlir::Value addr = buildMemAddr(builder);
    return builder.createLoad(loc, builder.createPtrBitcast(addr, ty));
  }

  // The register save area holds the six GP registers in its first 48 bytes;
  // `gp_offset` is the byte offset of the next available one.  The aggregate
  // is passed in registers while `gp_offset <= 48 - neededInt * 8`.
  mlir::Value gpOffsetP = builder.createGetMember(
      loc, builder.getPointerTo(vaFields[0]), valist, "gp_offset", 0);
  mlir::Value gpOffset = builder.createLoad(loc, gpOffsetP);
  mlir::Value limit =
      builder.getConstantInt(loc, gpOffset.getType(), 48 - neededInt * 8);
  mlir::Value inRegs =
      builder.createCompare(loc, cir::CmpOpKind::le, gpOffset, limit);

  // Select the argument slot: in-register (reg_save_area + gp_offset, then bump
  // gp_offset past the consumed registers) or in-memory (overflow area).
  mlir::Value addr =
      cir::TernaryOp::create(
          builder, loc, inRegs,
          /*trueBuilder=*/
          [&](mlir::OpBuilder &ob, mlir::Location l) {
            CIRBaseBuilderTy b(ob);
            mlir::Value regSaveArea = b.createLoad(
                l, b.createGetMember(l, b.getPointerTo(vaFields[3]), valist,
                                     "reg_save_area", 3));
            regSaveArea = b.createPtrBitcast(regSaveArea, byteTy);
            mlir::Value regAddr = b.createPtrStride(l, regSaveArea, gpOffset);
            mlir::Value bump =
                b.getConstantInt(l, gpOffset.getType(), neededInt * 8);
            b.createStore(l, b.createAdd(l, gpOffset, bump), gpOffsetP);
            cir::YieldOp::create(b, l, regAddr);
          },
          /*falseBuilder=*/
          [&](mlir::OpBuilder &ob, mlir::Location l) {
            CIRBaseBuilderTy b(ob);
            cir::YieldOp::create(b, l, buildMemAddr(b));
          })
          .getResult();

  return builder.createLoad(loc, builder.createPtrBitcast(addr, ty));
}

} // namespace

std::unique_ptr<TargetLoweringInfo> createX86_64TargetLoweringInfo() {
  return std::make_unique<X86_64TargetLoweringInfo>();
}

} // namespace cir
