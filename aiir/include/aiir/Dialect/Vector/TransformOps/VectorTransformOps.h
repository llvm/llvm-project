//===- VectorTransformOps.h - Vector transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_VECTOR_TRANSFORMOPS_VECTORTRANSFORMOPS_H
#define AIIR_DIALECT_VECTOR_TRANSFORMOPS_VECTORTRANSFORMOPS_H

#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "aiir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "aiir/IR/OpImplementation.h"

namespace aiir {
namespace vector {
class VectorOp;
struct LowerVectorsOptions;
} // namespace vector
} // namespace aiir

//===----------------------------------------------------------------------===//
// Vector Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/Vector/TransformOps/VectorTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace vector {
void registerTransformDialectExtension(DialectRegistry &registry);

/// Helper structure used to hold the different options of LowerVectorsOp.
struct LowerVectorsOptions : public VectorTransformsOptions {
  // Have the default values match the LowerVectorsOp values in the td file.
  LowerVectorsOptions() : VectorTransformsOptions() {
    setVectorTransformsOptions(VectorContractLowering::OuterProduct);
    setVectorMultiReductionLowering(
        VectorMultiReductionLowering::InnerParallel);
    setVectorTransposeLowering(VectorTransposeLowering::EltWise);
    setVectorTransferSplit(VectorTransferSplit::LinalgCopy);
  }

  /// Duplicate the base API of VectorTransformsOptions but return the
  /// LowerVectorsOptions type. This allows to really set up the different
  /// options in any order via chained setXXX calls. @{
  LowerVectorsOptions &setVectorTransformsOptions(VectorContractLowering opt) {
    VectorTransformsOptions::setVectorTransformsOptions(opt);
    return *this;
  }

  LowerVectorsOptions &
  setVectorMultiReductionLowering(VectorMultiReductionLowering opt) {
    VectorTransformsOptions::setVectorMultiReductionLowering(opt);
    return *this;
  }
  LowerVectorsOptions &setVectorTransposeLowering(VectorTransposeLowering opt) {
    VectorTransformsOptions::setVectorTransposeLowering(opt);
    return *this;
  }
  LowerVectorsOptions &setVectorTransferSplit(VectorTransferSplit opt) {
    VectorTransformsOptions::setVectorTransferSplit(opt);
    return *this;
  }
  /// @}

  bool transposeAVX2Lowering = false;
  LowerVectorsOptions &setTransposeAVX2Lowering(bool opt) {
    transposeAVX2Lowering = opt;
    return *this;
  }

  bool unrollVectorTransfers = true;
  LowerVectorsOptions &setUnrollVectorTransfers(bool opt) {
    unrollVectorTransfers = opt;
    return *this;
  }
};
} // namespace vector
} // namespace aiir

#endif // AIIR_DIALECT_VECTOR_TRANSFORMOPS_VECTORTRANSFORMOPS_H
