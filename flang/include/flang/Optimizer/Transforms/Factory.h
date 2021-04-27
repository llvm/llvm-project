//===-- Optimizer/Transforms/Factory.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Templates to generate more complex code patterns in transformation passes.
// In transformation passes, front-end information such as is available in
// lowering is not available.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_FACTORY_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_FACTORY_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
class Location;
class Value;
} // namespace mlir

namespace fir::factory {

/// Generate a character copy with optimized forms.
///
/// If the lengths are constant and equal, use load/store rather than a loop.
/// Otherwise, if the lengths are constant and the input is longer than the
/// output, generate a loop to move a truncated portion of the source to the
/// destination. Finally, if the lengths are runtime values or the destination
/// is longer than the source, move the entire source character and pad the
/// destination with spaces as needed.
template <typename B>
void genCharacterCopy(mlir::Value src, mlir::Value srcLen, mlir::Value dst,
                      mlir::Value dstLen, B &builder, mlir::Location loc) {
  auto srcTy =
      fir::dyn_cast_ptrEleTy(src.getType()).template cast<fir::CharacterType>();
  auto dstTy =
      fir::dyn_cast_ptrEleTy(dst.getType()).template cast<fir::CharacterType>();
  if (!srcLen && !dstLen && srcTy.getFKind() == dstTy.getFKind() &&
      srcTy.getLen() == dstTy.getLen()) {
    // same size, so just use load and store
    auto load = builder.template create<fir::LoadOp>(loc, src);
    builder.template create<fir::StoreOp>(loc, load, dst);
    return;
  }
  auto zero = builder.template create<mlir::ConstantIndexOp>(loc, 0);
  auto one = builder.template create<mlir::ConstantIndexOp>(loc, 1);
  auto toArrayTy = [&](fir::CharacterType ty) {
    return fir::ReferenceType::get(fir::SequenceType::get(
        fir::SequenceType::ShapeRef{fir::SequenceType::getUnknownExtent()},
        fir::CharacterType::getSingleton(ty.getContext(), ty.getFKind())));
  };
  auto toEleTy = [&](fir::ReferenceType ty) {
    auto seqTy = ty.getEleTy().cast<fir::SequenceType>();
    return seqTy.getEleTy().cast<fir::CharacterType>();
  };
  auto toCoorTy = [&](fir::ReferenceType ty) {
    return fir::ReferenceType::get(toEleTy(ty));
  };
  if (!srcLen && !dstLen && srcTy.getLen() >= dstTy.getLen()) {
    auto upper =
        builder.template create<mlir::ConstantIndexOp>(loc, dstTy.getLen() - 1);
    auto loop = builder.template create<fir::DoLoopOp>(loc, zero, upper, one);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());
    auto csrcTy = toArrayTy(srcTy);
    auto csrc = builder.template create<fir::ConvertOp>(loc, csrcTy, src);
    auto in = builder.template create<fir::CoordinateOp>(
        loc, toCoorTy(csrcTy), csrc, loop.getInductionVar());
    auto load = builder.template create<fir::LoadOp>(loc, in);
    auto cdstTy = toArrayTy(dstTy);
    auto cdst = builder.template create<fir::ConvertOp>(loc, cdstTy, dst);
    auto out = builder.template create<fir::CoordinateOp>(
        loc, toCoorTy(cdstTy), cdst, loop.getInductionVar());
    mlir::Value cast =
        srcTy.getFKind() == dstTy.getFKind()
            ? load.getResult()
            : builder
                  .template create<fir::ConvertOp>(loc, toEleTy(cdstTy), load)
                  .getResult();
    builder.template create<fir::StoreOp>(loc, cast, out);
    builder.restoreInsertionPoint(insPt);
    return;
  }
  auto minusOne = [&](mlir::Value v) -> mlir::Value {
    return builder.template create<mlir::SubIOp>(
        loc, builder.template create<fir::ConvertOp>(loc, one.getType(), v),
        one);
  };
  mlir::Value len =
      dstLen
          ? minusOne(dstLen)
          : builder
                .template create<mlir::ConstantIndexOp>(loc, dstTy.getLen() - 1)
                .getResult();
  auto loop = builder.template create<fir::DoLoopOp>(loc, zero, len, one);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());
  mlir::Value slen =
      srcLen
          ? builder.template create<fir::ConvertOp>(loc, one.getType(), srcLen)
                .getResult()
          : builder.template create<mlir::ConstantIndexOp>(loc, srcTy.getLen())
                .getResult();
  auto cond = builder.template create<mlir::CmpIOp>(
      loc, mlir::CmpIPredicate::slt, loop.getInductionVar(), slen);
  auto ifOp = builder.template create<fir::IfOp>(loc, cond, /*withElse=*/true);
  builder.setInsertionPointToStart(&ifOp.thenRegion().front());
  auto csrcTy = toArrayTy(srcTy);
  auto csrc = builder.template create<fir::ConvertOp>(loc, csrcTy, src);
  auto in = builder.template create<fir::CoordinateOp>(
      loc, toCoorTy(csrcTy), csrc, loop.getInductionVar());
  auto load = builder.template create<fir::LoadOp>(loc, in);
  auto cdstTy = toArrayTy(dstTy);
  auto cdst = builder.template create<fir::ConvertOp>(loc, cdstTy, dst);
  auto out = builder.template create<fir::CoordinateOp>(
      loc, toCoorTy(cdstTy), cdst, loop.getInductionVar());
  mlir::Value cast =
      srcTy.getFKind() == dstTy.getFKind()
          ? load.getResult()
          : builder.template create<fir::ConvertOp>(loc, toEleTy(cdstTy), load)
                .getResult();
  builder.template create<fir::StoreOp>(loc, cast, out);
  builder.setInsertionPointToStart(&ifOp.elseRegion().front());
  auto space = builder.template create<fir::StringLitOp>(
      loc, toEleTy(cdstTy), llvm::ArrayRef<char>{' '});
  auto cdst2 = builder.template create<fir::ConvertOp>(loc, cdstTy, dst);
  auto out2 = builder.template create<fir::CoordinateOp>(
      loc, toCoorTy(cdstTy), cdst2, loop.getInductionVar());
  builder.template create<fir::StoreOp>(loc, space, out2);
  builder.restoreInsertionPoint(insPt);
}

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_FACTORY_H
