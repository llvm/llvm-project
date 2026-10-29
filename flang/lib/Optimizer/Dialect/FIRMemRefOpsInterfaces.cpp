//===-- FIRMemRefOpsInterfaces.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// External models of FortranObjectViewOpInterface for memref casts, so that
/// FIR alias analysis can look through them. fir-to-memref creates
/// memref.reinterpret_cast, and canonicalization can also introduce
/// memref.cast.
///
/// We assume the result stays within the object of the source, as we do
/// elsewhere in FIR alias analysis.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include <optional>

namespace {

/// Must be specialized for each operation to state its view offset.
template <typename OP>
struct MemRefViewModel
    : public fir::FortranObjectViewOpInterface::ExternalModel<
          MemRefViewModel<OP>, OP> {
  mlir::Value getViewSource(mlir::Operation *op, mlir::OpResult) const {
    return mlir::cast<mlir::ViewLikeOpInterface>(op).getViewSource();
  }
  std::optional<std::int64_t> getViewOffset(mlir::Operation *op,
                                            mlir::OpResult resultView) const;
};

template <>
std::optional<std::int64_t>
MemRefViewModel<mlir::memref::CastOp>::getViewOffset(mlir::Operation *,
                                                     mlir::OpResult) const {
  // Address is the same as the source.
  return 0;
}

template <>
std::optional<std::int64_t>
MemRefViewModel<mlir::memref::ReinterpretCastOp>::getViewOffset(
    mlir::Operation *op, mlir::OpResult) const {
  auto cast = mlir::cast<mlir::memref::ReinterpretCastOp>(op);

  llvm::SmallVector<std::int64_t> strides;
  std::int64_t srcOffset;

  // Address is only the same as the source if both offsets are zero, since the
  // offset is relative to the base buffer.

  // If the source is unranked there is no static offset.
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
  if (srcTy && mlir::succeeded(srcTy.getStridesAndOffset(strides, srcOffset)) &&
      srcOffset == 0) {
    // The result offset can be a constant operand.
    std::optional<std::int64_t> resOffset =
        mlir::getConstantIntValue(cast.getMixedOffsets()[0]);
    if (resOffset == 0)
      return 0;
  }

  return std::nullopt;
}

} // namespace

void fir::registerMemRefFortranObjectViewExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::memref::MemRefDialect *dialect) {
        mlir::memref::CastOp::attachInterface<
            MemRefViewModel<mlir::memref::CastOp>>(*ctx);
        mlir::memref::ReinterpretCastOp::attachInterface<
            MemRefViewModel<mlir::memref::ReinterpretCastOp>>(*ctx);
      });
}
