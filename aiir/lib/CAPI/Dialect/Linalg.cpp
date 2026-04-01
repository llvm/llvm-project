//===- Linalg.cpp - C Interface for Linalg dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Linalg.h"
#include "aiir/CAPI/AffineMap.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"

using namespace aiir;
using namespace aiir::linalg;

/// Apply the special region builder for the builtin named Linalg op.
/// Assert that `op` is a builtin named Linalg op.
void aiirLinalgFillBuiltinNamedOpRegion(AiirOperation aiirOp) {
  Operation *op = unwrap(aiirOp);
  auto linalgOp = cast<LinalgOp>(op);
  auto *dialect = static_cast<LinalgDialect *>(linalgOp->getDialect());
  LinalgDialect::RegionBuilderFunType fun =
      dialect->getRegionBuilder(op->getName().getStringRef());

  assert(fun && "Expected a builtin named Linalg op.");
  assert(op->getNumRegions() == 1 && "Expected Linalg op with 1 region");
  assert(op->getRegion(0).getBlocks().empty() &&
         "Expected Linalg op with 0 blocks");

  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    argTypes.push_back(getElementTypeOrSelf(opOperand.get().getType()));
    argLocs.push_back(opOperand.get().getLoc());
  }

  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  b.setInsertionPointToStart(body);
  fun(b, *body, op->getAttrs(), /*emitError=*/{});
}

AIIR_CAPI_EXPORTED bool aiirLinalgIsAContractionOp(AiirOperation op) {
  auto linalgOp = llvm::dyn_cast<aiir::linalg::LinalgOp>(unwrap(op));
  // isaContractionOpInterface handles null linalgOp internally.
  return linalg::isaContractionOpInterface(linalgOp);
}

AIIR_CAPI_EXPORTED AiirLinalgContractionDimensions
aiirLinalgInferContractionDimensions(AiirOperation op) {
  AiirLinalgContractionDimensions result{};
  auto linalgOp = dyn_cast<linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return result;

  FailureOr<linalg::ContractionDimensions> maybeDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(maybeDims))
    return result;

  const linalg::ContractionDimensions &contractionDims = *maybeDims;
  AIIRContext *ctx = linalgOp.getContext();

  auto toAttr = [ctx](ArrayRef<unsigned> vals) -> AiirAttribute {
    return wrap(DenseI32ArrayAttr::get(ctx, llvm::to_vector_of<int32_t>(vals)));
  };

  result.batch = toAttr(contractionDims.batch);
  result.m = toAttr(contractionDims.m);
  result.n = toAttr(contractionDims.n);
  result.k = toAttr(contractionDims.k);

  return result;
}

AIIR_CAPI_EXPORTED AiirLinalgContractionDimensions
aiirLinalgInferContractionDimensionsFromMaps(const AiirAffineMap *indexingMaps,
                                             size_t numMaps) {
  AiirLinalgContractionDimensions result{};
  if (!indexingMaps || numMaps == 0)
    return result;

  SmallVector<AffineMap, 3> maps;
  maps.reserve(numMaps);
  for (size_t i = 0; i < numMaps; ++i) {
    maps.push_back(unwrap(indexingMaps[i]));
  }

  FailureOr<linalg::ContractionDimensions> maybeDims =
      linalg::inferContractionDims(maps);
  if (failed(maybeDims))
    return result;

  AIIRContext *ctx = maps[0].getContext();

  auto toAttr = [ctx](ArrayRef<unsigned> vals) -> AiirAttribute {
    return wrap(DenseI32ArrayAttr::get(ctx, llvm::to_vector_of<int32_t>(vals)));
  };

  result.batch = toAttr(maybeDims->batch);
  result.m = toAttr(maybeDims->m);
  result.n = toAttr(maybeDims->n);
  result.k = toAttr(maybeDims->k);

  return result;
}

AIIR_CAPI_EXPORTED bool aiirLinalgIsAConvolutionOp(AiirOperation op) {
  auto linalgOp = llvm::dyn_cast<aiir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return false;

  return linalg::isaConvolutionOpInterface(linalgOp);
}

AIIR_CAPI_EXPORTED AiirLinalgConvolutionDimensions
aiirLinalgInferConvolutionDimensions(AiirOperation op) {
  AiirLinalgConvolutionDimensions result{};
  auto linalgOp = llvm::dyn_cast<aiir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return result;

  FailureOr<linalg::ConvolutionDimensions> maybeDims =
      linalg::inferConvolutionDims(linalgOp);
  if (failed(maybeDims))
    return result;

  const linalg::ConvolutionDimensions &dims = *maybeDims;
  AIIRContext *ctx = linalgOp.getContext();

  auto toI32Attr =
      [&ctx](const SmallVector<unsigned, 2> &vals) -> AiirAttribute {
    return wrap(DenseI32ArrayAttr::get(ctx, llvm::to_vector_of<int32_t>(vals)));
  };

  auto toI64Attr =
      [&ctx](const SmallVector<int64_t, 2> &vals) -> AiirAttribute {
    return wrap(DenseI64ArrayAttr::get(ctx, vals));
  };

  result.batch = toI32Attr(dims.batch);
  result.outputImage = toI32Attr(dims.outputImage);
  result.outputChannel = toI32Attr(dims.outputChannel);
  result.filterLoop = toI32Attr(dims.filterLoop);
  result.inputChannel = toI32Attr(dims.inputChannel);
  result.depth = toI32Attr(dims.depth);
  result.strides = toI64Attr(dims.strides);
  result.dilations = toI64Attr(dims.dilations);

  return result;
}

AIIR_CAPI_EXPORTED AiirAttribute
aiirLinalgGetIndexingMapsAttribute(AiirOperation op) {
  auto linalgOp = llvm::dyn_cast<aiir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return AiirAttribute{nullptr};

  ArrayAttr attr = linalgOp.getIndexingMaps();
  return wrap(attr);
}

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linalg, linalg, LinalgDialect)
