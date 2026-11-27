//===- DescriptorScalarization.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Utils/DescriptorScalarization.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace fir::acc {

fir::SequenceType getScalarizableArrayType(fir::BaseBoxType boxType) {
  if (isa<fir::ClassType>(boxType))
    return {};

  auto arrayType =
      dyn_cast<fir::SequenceType>(boxType.getElementOrSequenceType());
  // Cannot scalarize descriptors without a statically known array rank.
  if (!arrayType || arrayType.hasUnknownShape())
    return {};

  Type elementType = arrayType.getEleTy();
  // Cannot scalarize descriptors of dynamic-length characters or derived types
  // with LEN parameters.
  if (fir::characterWithDynamicLen(elementType) ||
      fir::isRecordWithTypeParameters(elementType))
    return {};

  return arrayType;
}

bool mayRequireBaseAddress(Operation *op) {
  return !isa<fir::BoxDimsOp, fir::BoxRankOp, fir::BoxEleSizeOp,
              fir::BoxTypeCodeOp, fir::BoxIsAllocOp, fir::BoxIsPtrOp,
              fir::BoxIsArrayOp, fir::IsAssumedSizeOp, fir::IsContiguousBoxOp,
              fir::BoxTotalElementsOp>(op);
}

ShapeScalars emitShapeScalars(OpBuilder &builder, Location loc, Value box,
                              unsigned rank) {
  Type indexType = builder.getIndexType();
  ShapeScalars shape;
  for (unsigned i = 0; i < rank; ++i) {
    Value dim = arith::ConstantIndexOp::create(builder, loc, i);
    auto dims = fir::BoxDimsOp::create(builder, loc, indexType, indexType,
                                       indexType, box, dim);
    shape.lbs.push_back(dims.getLowerBound());
    shape.extents.push_back(dims.getExtent());
    shape.strides.push_back(dims.getByteStride());
  }
  return shape;
}

} // namespace fir::acc
