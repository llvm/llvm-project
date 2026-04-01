//===- quant.c - Test of Quant dialect C API ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-quant-test 2>&1 | FileCheck %s

#include "aiir-c/Dialect/Quant.h"
#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/IR.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testTypeHierarchy
static void testTypeHierarchy(AiirContext ctx) {
  fprintf(stderr, "testTypeHierarchy\n");

  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType any = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>"));
  AiirType uniform =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(
                                "!quant.uniform<i8<-8:7>:f32, 0.99872:127>"));
  AiirType perAxis = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString(
               "!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>"));
  AiirType calibrated = aiirTypeParseGet(
      ctx,
      aiirStringRefCreateFromCString("!quant.calibrated<f32<-0.998:1.2321>>"));

  // The parser itself is checked in C++ dialect tests.
  assert(!aiirTypeIsNull(any) && "couldn't parse AnyQuantizedType");
  assert(!aiirTypeIsNull(uniform) && "couldn't parse UniformQuantizedType");
  assert(!aiirTypeIsNull(perAxis) &&
         "couldn't parse UniformQuantizedPerAxisType");
  assert(!aiirTypeIsNull(calibrated) &&
         "couldn't parse CalibratedQuantizedType");

  // CHECK: i8 isa QuantizedType: 0
  fprintf(stderr, "i8 isa QuantizedType: %d\n", aiirTypeIsAQuantizedType(i8));
  // CHECK: any isa QuantizedType: 1
  fprintf(stderr, "any isa QuantizedType: %d\n", aiirTypeIsAQuantizedType(any));
  // CHECK: uniform isa QuantizedType: 1
  fprintf(stderr, "uniform isa QuantizedType: %d\n",
          aiirTypeIsAQuantizedType(uniform));
  // CHECK: perAxis isa QuantizedType: 1
  fprintf(stderr, "perAxis isa QuantizedType: %d\n",
          aiirTypeIsAQuantizedType(perAxis));
  // CHECK: calibrated isa QuantizedType: 1
  fprintf(stderr, "calibrated isa QuantizedType: %d\n",
          aiirTypeIsAQuantizedType(calibrated));

  // CHECK: any isa AnyQuantizedType: 1
  fprintf(stderr, "any isa AnyQuantizedType: %d\n",
          aiirTypeIsAAnyQuantizedType(any));
  // CHECK: uniform isa UniformQuantizedType: 1
  fprintf(stderr, "uniform isa UniformQuantizedType: %d\n",
          aiirTypeIsAUniformQuantizedType(uniform));
  // CHECK: perAxis isa UniformQuantizedPerAxisType: 1
  fprintf(stderr, "perAxis isa UniformQuantizedPerAxisType: %d\n",
          aiirTypeIsAUniformQuantizedPerAxisType(perAxis));
  // CHECK: calibrated isa CalibratedQuantizedType: 1
  fprintf(stderr, "calibrated isa CalibratedQuantizedType: %d\n",
          aiirTypeIsACalibratedQuantizedType(calibrated));

  // CHECK: perAxis isa UniformQuantizedType: 0
  fprintf(stderr, "perAxis isa UniformQuantizedType: %d\n",
          aiirTypeIsAUniformQuantizedType(perAxis));
  // CHECK: uniform isa CalibratedQuantizedType: 0
  fprintf(stderr, "uniform isa CalibratedQuantizedType: %d\n",
          aiirTypeIsACalibratedQuantizedType(uniform));
  fprintf(stderr, "\n");
}

// CHECK-LABEL: testAnyQuantizedType
void testAnyQuantizedType(AiirContext ctx) {
  fprintf(stderr, "testAnyQuantizedType\n");

  AiirType anyParsed = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>"));

  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType f32 = aiirF32TypeGet(ctx);
  AiirType any =
      aiirAnyQuantizedTypeGet(aiirQuantizedTypeGetSignedFlag(), i8, f32, -8, 7);

  // CHECK: flags: 1
  fprintf(stderr, "flags: %u\n", aiirQuantizedTypeGetFlags(any));
  // CHECK: signed: 1
  fprintf(stderr, "signed: %u\n", aiirQuantizedTypeIsSigned(any));
  // CHECK: storage type: i8
  fprintf(stderr, "storage type: ");
  aiirTypeDump(aiirQuantizedTypeGetStorageType(any));
  fprintf(stderr, "\n");
  // CHECK: expressed type: f32
  fprintf(stderr, "expressed type: ");
  aiirTypeDump(aiirQuantizedTypeGetExpressedType(any));
  fprintf(stderr, "\n");
  // CHECK: storage min: -8
  fprintf(stderr, "storage min: %" PRId64 "\n",
          aiirQuantizedTypeGetStorageTypeMin(any));
  // CHECK: storage max: 7
  fprintf(stderr, "storage max: %" PRId64 "\n",
          aiirQuantizedTypeGetStorageTypeMax(any));
  // CHECK: storage width: 8
  fprintf(stderr, "storage width: %u\n",
          aiirQuantizedTypeGetStorageTypeIntegralWidth(any));
  // CHECK: quantized element type: !quant.any<i8<-8:7>:f32>
  fprintf(stderr, "quantized element type: ");
  aiirTypeDump(aiirQuantizedTypeGetQuantizedElementType(any));
  fprintf(stderr, "\n");

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(anyParsed, any));
  // CHECK: !quant.any<i8<-8:7>:f32>
  aiirTypeDump(any);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testUniformType
void testUniformType(AiirContext ctx) {
  fprintf(stderr, "testUniformType\n");

  AiirType uniformParsed =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(
                                "!quant.uniform<i8<-8:7>:f32, 0.99872:127>"));

  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType f32 = aiirF32TypeGet(ctx);
  AiirType uniform = aiirUniformQuantizedTypeGet(
      aiirQuantizedTypeGetSignedFlag(), i8, f32, 0.99872, 127, -8, 7);

  // CHECK: scale: 0.998720
  fprintf(stderr, "scale: %lf\n", aiirUniformQuantizedTypeGetScale(uniform));
  // CHECK: zero point: 127
  fprintf(stderr, "zero point: %" PRId64 "\n",
          aiirUniformQuantizedTypeGetZeroPoint(uniform));
  // CHECK: fixed point: 0
  fprintf(stderr, "fixed point: %d\n",
          aiirUniformQuantizedTypeIsFixedPoint(uniform));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(uniform, uniformParsed));
  // CHECK: !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
  aiirTypeDump(uniform);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testUniformPerAxisType
void testUniformPerAxisType(AiirContext ctx) {
  fprintf(stderr, "testUniformPerAxisType\n");

  AiirType perAxisParsed = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString(
               "!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>"));

  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType f32 = aiirF32TypeGet(ctx);
  double scales[] = {200.0, 0.99872};
  int64_t zeroPoints[] = {0, 120};
  AiirType perAxis = aiirUniformQuantizedPerAxisTypeGet(
      aiirQuantizedTypeGetSignedFlag(), i8, f32,
      /*nDims=*/2, scales, zeroPoints,
      /*quantizedDimension=*/1,
      aiirQuantizedTypeGetDefaultMinimumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8),
      aiirQuantizedTypeGetDefaultMaximumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8));

  // CHECK: num dims: 2
  fprintf(stderr, "num dims: %" PRIdPTR "\n",
          aiirUniformQuantizedPerAxisTypeGetNumDims(perAxis));
  // CHECK: scale 0: 200.000000
  fprintf(stderr, "scale 0: %lf\n",
          aiirUniformQuantizedPerAxisTypeGetScale(perAxis, 0));
  // CHECK: scale 1: 0.998720
  fprintf(stderr, "scale 1: %lf\n",
          aiirUniformQuantizedPerAxisTypeGetScale(perAxis, 1));
  // CHECK: zero point 0: 0
  fprintf(stderr, "zero point 0: %" PRId64 "\n",
          aiirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 0));
  // CHECK: zero point 1: 120
  fprintf(stderr, "zero point 1: %" PRId64 "\n",
          aiirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 1));
  // CHECK: quantized dim: 1
  fprintf(stderr, "quantized dim: %" PRId32 "\n",
          aiirUniformQuantizedPerAxisTypeGetQuantizedDimension(perAxis));
  // CHECK: fixed point: 0
  fprintf(stderr, "fixed point: %d\n",
          aiirUniformQuantizedPerAxisTypeIsFixedPoint(perAxis));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(perAxis, perAxisParsed));
  // CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>
  aiirTypeDump(perAxis);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testUniformSubChannelType
void testUniformSubChannelType(AiirContext ctx) {
  fprintf(stderr, "testUniformSubChannelType\n");

  AiirType subChannelParsed =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(
                                "!quant.uniform<i8:f32:{0:1, 1:2}, "
                                "{{2.0:10, 3.0:20}, {4.0:30, 5.0:40}}>"));

  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType f32 = aiirF32TypeGet(ctx);

  // block-size information
  int32_t quantizedDimensions[] = {0, 1};
  int64_t blockSizes[] = {1, 2};
  int64_t numBlockSizes = 2;

  // quantization parameters
  int64_t quantParamShape[] = {2, 2};
  int64_t quantParamRank = 2;
  int64_t numQuantizationParams = 4;
  AiirAttribute scales[] = {aiirFloatAttrDoubleGet(ctx, f32, 2.0),
                            aiirFloatAttrDoubleGet(ctx, f32, 3.0),
                            aiirFloatAttrDoubleGet(ctx, f32, 4.0),
                            aiirFloatAttrDoubleGet(ctx, f32, 5.0)};
  AiirAttribute zeroPoints[] = {
      aiirIntegerAttrGet(i8, 10), aiirIntegerAttrGet(i8, 20),
      aiirIntegerAttrGet(i8, 30), aiirIntegerAttrGet(i8, 40)};

  AiirType scalesType =
      aiirRankedTensorTypeGet(quantParamRank, quantParamShape, f32,
                              /*encoding=*/aiirAttributeGetNull());
  AiirType zeroPointsType = aiirRankedTensorTypeGet(
      quantParamRank, quantParamShape, i8, /*encoding=*/aiirAttributeGetNull());
  AiirAttribute denseScalesAttr =
      aiirDenseElementsAttrGet(scalesType, numQuantizationParams, scales);
  AiirAttribute denseZeroPointsAttr = aiirDenseElementsAttrGet(
      zeroPointsType, numQuantizationParams, zeroPoints);

  AiirType subChannel = aiirUniformQuantizedSubChannelTypeGet(
      aiirQuantizedTypeGetSignedFlag(), i8, f32, denseScalesAttr,
      denseZeroPointsAttr, numBlockSizes, quantizedDimensions, blockSizes,
      aiirQuantizedTypeGetDefaultMinimumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8),
      aiirQuantizedTypeGetDefaultMaximumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8));

  AiirAttribute arrayScalesAttr =
      aiirArrayAttrGet(ctx, numQuantizationParams, scales);
  AiirAttribute arrayZeroPointsAttr =
      aiirArrayAttrGet(ctx, numQuantizationParams, zeroPoints);
  AiirType illegalSubChannel = aiirUniformQuantizedSubChannelTypeGet(
      aiirQuantizedTypeGetSignedFlag(), i8, f32, arrayScalesAttr,
      arrayZeroPointsAttr, numBlockSizes, quantizedDimensions, blockSizes,
      aiirQuantizedTypeGetDefaultMinimumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8),
      aiirQuantizedTypeGetDefaultMaximumForInteger(/*isSigned=*/true,
                                                   /*integralWidth=*/8));

  // CHECK: is null sub-channel type: 1
  fprintf(stderr, "is null sub-channel type: %d\n",
          aiirTypeIsNull(illegalSubChannel));

  // CHECK: num dims: 2
  fprintf(stderr, "num dims: %" PRIdPTR "\n",
          aiirUniformQuantizedSubChannelTypeGetNumBlockSizes(subChannel));

  // CHECK: axis-block-size-pair[0]: 0:1
  fprintf(
      stderr, "axis-block-size-pair[0]: %" PRId32 ":%" PRId64 "\n",
      aiirUniformQuantizedSubChannelTypeGetQuantizedDimension(subChannel, 0),
      aiirUniformQuantizedSubChannelTypeGetBlockSize(subChannel, 0));

  // CHECK: axis-block-size-pair[1]: 1:2
  fprintf(
      stderr, "axis-block-size-pair[1]: %" PRId32 ":%" PRId64 "\n",
      aiirUniformQuantizedSubChannelTypeGetQuantizedDimension(subChannel, 1),
      aiirUniformQuantizedSubChannelTypeGetBlockSize(subChannel, 1));

  denseScalesAttr = aiirUniformQuantizedSubChannelTypeGetScales(subChannel);
  denseZeroPointsAttr =
      aiirUniformQuantizedSubChannelTypeGetZeroPoints(subChannel);
  scalesType = aiirAttributeGetType(denseScalesAttr);
  zeroPointsType = aiirAttributeGetType(denseZeroPointsAttr);

  // CHECK: tensor<2x2xf32>
  aiirTypeDump(scalesType);
  // CHECK: tensor<2x2xi8>
  aiirTypeDump(zeroPointsType);

  // CHECK: number of quantization parameters: 4
  fprintf(stderr, "number of quantization parameters: %" PRId64 "\n",
          aiirElementsAttrGetNumElements(denseScalesAttr));

  // CHECK: quantization-parameter[0]: 2.000000:10
  fprintf(stderr, "quantization-parameter[0]: %lf:%" PRId8 "\n",
          aiirDenseElementsAttrGetFloatValue(denseScalesAttr, 0),
          aiirDenseElementsAttrGetInt8Value(denseZeroPointsAttr, 0));

  // CHECK: quantization-parameter[1]: 3.000000:20
  fprintf(stderr, "quantization-parameter[1]: %lf:%" PRId8 "\n",
          aiirDenseElementsAttrGetFloatValue(denseScalesAttr, 1),
          aiirDenseElementsAttrGetInt8Value(denseZeroPointsAttr, 1));

  // CHECK: quantization-parameter[2]: 4.000000:30
  fprintf(stderr, "quantization-parameter[2]: %lf:%" PRId8 "\n",
          aiirDenseElementsAttrGetFloatValue(denseScalesAttr, 2),
          aiirDenseElementsAttrGetInt8Value(denseZeroPointsAttr, 2));

  // CHECK: quantization-parameter[3]: 5.000000:40
  fprintf(stderr, "quantization-parameter[3]: %lf:%" PRId8 "\n",
          aiirDenseElementsAttrGetFloatValue(denseScalesAttr, 3),
          aiirDenseElementsAttrGetInt8Value(denseZeroPointsAttr, 3));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(subChannel, subChannelParsed));

  // CHECK: !quant.uniform<i8:f32:{0:1, 1:2},
  // {{.*}}2.000000e+00:10, 3.000000e+00:20},
  // {4.000000e+00:30, 5.000000e+00:40{{.*}}}}>
  aiirTypeDump(subChannel);
  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testCalibratedType
void testCalibratedType(AiirContext ctx) {
  fprintf(stderr, "testCalibratedType\n");

  AiirType calibratedParsed = aiirTypeParseGet(
      ctx,
      aiirStringRefCreateFromCString("!quant.calibrated<f32<-0.998:1.2321>>"));

  AiirType f32 = aiirF32TypeGet(ctx);
  AiirType calibrated = aiirCalibratedQuantizedTypeGet(f32, -0.998, 1.2321);

  // CHECK: min: -0.998000
  fprintf(stderr, "min: %lf\n", aiirCalibratedQuantizedTypeGetMin(calibrated));
  // CHECK: max: 1.232100
  fprintf(stderr, "max: %lf\n", aiirCalibratedQuantizedTypeGetMax(calibrated));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(calibrated, calibratedParsed));
  // CHECK: !quant.calibrated<f32<-0.998:1.232100e+00>>
  aiirTypeDump(calibrated);
  fprintf(stderr, "\n\n");
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__quant__(), ctx);
  testTypeHierarchy(ctx);
  testAnyQuantizedType(ctx);
  testUniformType(ctx);
  testUniformPerAxisType(ctx);
  testUniformSubChannelType(ctx);
  testCalibratedType(ctx);
  aiirContextDestroy(ctx);
  return EXIT_SUCCESS;
}
