//===- DescriptorScalarization.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for descriptor scalarization, which involves passing descriptor
// fields as scalar inputs to an offload region and rebuilding the descriptor
// inside it.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENACC_UTILS_DESCRIPTORSCALARIZATION_H
#define FORTRAN_OPTIMIZER_OPENACC_UTILS_DESCRIPTORSCALARIZATION_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Location;
class OpBuilder;
class Operation;
} // namespace mlir

namespace fir::acc {

/// Returns the array type if the descriptor can be scalarized with
/// fir.create_box, or a null type otherwise.
fir::SequenceType getScalarizableArrayType(fir::BaseBoxType boxType);

/// Returns true if the operation may require a valid descriptor data base
/// address.
bool mayRequireBaseAddress(mlir::Operation *op);

/// Per-dimension lower bounds, extents, and byte strides of a descriptor.
struct ShapeScalars {
  llvm::SmallVector<mlir::Value> lbs, extents, strides;
};

/// Emit fir.box_dims for each dimension at the builder's insertion point.
ShapeScalars emitShapeScalars(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value box, unsigned rank);

} // namespace fir::acc

#endif // FORTRAN_OPTIMIZER_OPENACC_UTILS_DESCRIPTORSCALARIZATION_H
