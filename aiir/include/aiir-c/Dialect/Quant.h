//===-- aiir-c/Dialect/Quant.h - C API for LLVM -------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_QUANT_H
#define AIIR_C_DIALECT_QUANT_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(quant, quant);

//===---------------------------------------------------------------------===//
// QuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a quantization dialect type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAQuantizedType(AiirType type);

/// Returns the bit flag used to indicate signedness of a quantized type.
AIIR_CAPI_EXPORTED unsigned aiirQuantizedTypeGetSignedFlag(void);

/// Returns the minimum possible value stored by a quantized type.
AIIR_CAPI_EXPORTED int64_t aiirQuantizedTypeGetDefaultMinimumForInteger(
    bool isSigned, unsigned integralWidth);

/// Returns the maximum possible value stored by a quantized type.
AIIR_CAPI_EXPORTED int64_t aiirQuantizedTypeGetDefaultMaximumForInteger(
    bool isSigned, unsigned integralWidth);

/// Gets the original type approximated by the given quantized type.
AIIR_CAPI_EXPORTED AiirType aiirQuantizedTypeGetExpressedType(AiirType type);

/// Gets the flags associated with the given quantized type.
AIIR_CAPI_EXPORTED unsigned aiirQuantizedTypeGetFlags(AiirType type);

/// Returns `true` if the given type is signed, `false` otherwise.
AIIR_CAPI_EXPORTED bool aiirQuantizedTypeIsSigned(AiirType type);

/// Returns the underlying type used to store the values.
AIIR_CAPI_EXPORTED AiirType aiirQuantizedTypeGetStorageType(AiirType type);

/// Returns the minimum value that the storage type of the given quantized type
/// can take.
AIIR_CAPI_EXPORTED int64_t aiirQuantizedTypeGetStorageTypeMin(AiirType type);

/// Returns the maximum value that the storage type of the given quantized type
/// can take.
AIIR_CAPI_EXPORTED int64_t aiirQuantizedTypeGetStorageTypeMax(AiirType type);

/// Returns the integral bitwidth that the storage type of the given quantized
/// type can represent exactly.
AIIR_CAPI_EXPORTED unsigned
aiirQuantizedTypeGetStorageTypeIntegralWidth(AiirType type);

/// Returns `true` if the `candidate` type is compatible with the given
/// quantized `type`.
AIIR_CAPI_EXPORTED bool
aiirQuantizedTypeIsCompatibleExpressedType(AiirType type, AiirType candidate);

/// Returns the element type of the given quantized type as another quantized
/// type.
AIIR_CAPI_EXPORTED AiirType
aiirQuantizedTypeGetQuantizedElementType(AiirType type);

/// Casts from a type based on the storage type of the given type to a
/// corresponding type based on the given type. Returns a null type if the cast
/// is not valid.
AIIR_CAPI_EXPORTED AiirType
aiirQuantizedTypeCastFromStorageType(AiirType type, AiirType candidate);

/// Casts from a type based on a quantized type to a corresponding typed based
/// on the storage type. Returns a null type if the cast is not valid.
AIIR_CAPI_EXPORTED AiirType aiirQuantizedTypeCastToStorageType(AiirType type);

/// Casts from a type based on the expressed type of the given type to a
/// corresponding type based on the given type. Returns a null type if the cast
/// is not valid.
AIIR_CAPI_EXPORTED AiirType
aiirQuantizedTypeCastFromExpressedType(AiirType type, AiirType candidate);

/// Casts from a type based on a quantized type to a corresponding typed based
/// on the expressed type. Returns a null type if the cast is not valid.
AIIR_CAPI_EXPORTED AiirType aiirQuantizedTypeCastToExpressedType(AiirType type);

/// Casts from a type based on the expressed type of the given quantized type to
/// equivalent type based on storage type of the same quantized type.
AIIR_CAPI_EXPORTED AiirType
aiirQuantizedTypeCastExpressedToStorageType(AiirType type, AiirType candidate);

//===---------------------------------------------------------------------===//
// AnyQuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is an AnyQuantizedType.
AIIR_CAPI_EXPORTED bool aiirTypeIsAAnyQuantizedType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirAnyQuantizedTypeGetTypeID(void);

/// Creates an instance of AnyQuantizedType with the given parameters in the
/// same context as `storageType` and returns it. The instance is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirAnyQuantizedTypeGet(unsigned flags,
                                                    AiirType storageType,
                                                    AiirType expressedType,
                                                    int64_t storageTypeMin,
                                                    int64_t storageTypeMax);

AIIR_CAPI_EXPORTED AiirStringRef aiirAnyQuantizedTypeGetName(void);

//===---------------------------------------------------------------------===//
// UniformQuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a UniformQuantizedType.
AIIR_CAPI_EXPORTED bool aiirTypeIsAUniformQuantizedType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirUniformQuantizedTypeGetTypeID(void);

/// Creates an instance of UniformQuantizedType with the given parameters in the
/// same context as `storageType` and returns it. The instance is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirUniformQuantizedTypeGet(
    unsigned flags, AiirType storageType, AiirType expressedType, double scale,
    int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax);

AIIR_CAPI_EXPORTED AiirStringRef aiirUniformQuantizedTypeGetName(void);

/// Returns the scale of the given uniform quantized type.
AIIR_CAPI_EXPORTED double aiirUniformQuantizedTypeGetScale(AiirType type);

/// Returns the zero point of the given uniform quantized type.
AIIR_CAPI_EXPORTED int64_t aiirUniformQuantizedTypeGetZeroPoint(AiirType type);

/// Returns `true` if the given uniform quantized type is fixed-point.
AIIR_CAPI_EXPORTED bool aiirUniformQuantizedTypeIsFixedPoint(AiirType type);

//===---------------------------------------------------------------------===//
// UniformQuantizedPerAxisType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a UniformQuantizedPerAxisType.
AIIR_CAPI_EXPORTED bool aiirTypeIsAUniformQuantizedPerAxisType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirUniformQuantizedPerAxisTypeGetTypeID(void);

/// Creates an instance of UniformQuantizedPerAxisType with the given parameters
/// in the same context as `storageType` and returns it. `scales` and
/// `zeroPoints` point to `nDims` number of elements. The instance is owned
/// by the context.
AIIR_CAPI_EXPORTED AiirType aiirUniformQuantizedPerAxisTypeGet(
    unsigned flags, AiirType storageType, AiirType expressedType,
    intptr_t nDims, double *scales, int64_t *zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin, int64_t storageTypeMax);

AIIR_CAPI_EXPORTED AiirStringRef aiirUniformQuantizedPerAxisTypeGetName(void);

/// Returns the number of axes in the given quantized per-axis type.
AIIR_CAPI_EXPORTED intptr_t
aiirUniformQuantizedPerAxisTypeGetNumDims(AiirType type);

/// Returns `pos`-th scale of the given quantized per-axis type.
AIIR_CAPI_EXPORTED double aiirUniformQuantizedPerAxisTypeGetScale(AiirType type,
                                                                  intptr_t pos);

/// Returns `pos`-th zero point of the given quantized per-axis type.
AIIR_CAPI_EXPORTED int64_t
aiirUniformQuantizedPerAxisTypeGetZeroPoint(AiirType type, intptr_t pos);

/// Returns the index of the quantized dimension in the given quantized per-axis
/// type.
AIIR_CAPI_EXPORTED int32_t
aiirUniformQuantizedPerAxisTypeGetQuantizedDimension(AiirType type);

/// Returns `true` if the given uniform quantized per-axis type is fixed-point.
AIIR_CAPI_EXPORTED bool
aiirUniformQuantizedPerAxisTypeIsFixedPoint(AiirType type);

//===---------------------------------------------------------------------===//
// UniformQuantizedSubChannelType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a UniformQuantizedSubChannel.
AIIR_CAPI_EXPORTED bool
aiirTypeIsAUniformQuantizedSubChannelType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirUniformQuantizedSubChannelTypeGetTypeID(void);

/// Creates a UniformQuantizedSubChannelType with the given parameters.
///
/// The type is owned by the context. `scalesAttr` and `zeroPointsAttr` must be
/// DenseElementsAttrs.  `quantizedDimensions` and `blockSizes`
/// point to `blockSizeInfoLength` number of elements, describing respectively
/// the quantization axis and corresponding block size.
AIIR_CAPI_EXPORTED AiirType aiirUniformQuantizedSubChannelTypeGet(
    unsigned flags, AiirType storageType, AiirType expressedType,
    AiirAttribute scalesAttr, AiirAttribute zeroPointsAttr,
    intptr_t blockSizeInfoLength, int32_t *quantizedDimensions,
    int64_t *blockSizes, int64_t storageTypeMin, int64_t storageTypeMax);

AIIR_CAPI_EXPORTED AiirStringRef
aiirUniformQuantizedSubChannelTypeGetName(void);

/// Returns the number of block sizes provided in type.
AIIR_CAPI_EXPORTED intptr_t
aiirUniformQuantizedSubChannelTypeGetNumBlockSizes(AiirType type);

/// Returns the quantized dimension at the given position.
AIIR_CAPI_EXPORTED int32_t
aiirUniformQuantizedSubChannelTypeGetQuantizedDimension(AiirType type,
                                                        intptr_t pos);

/// Returns the block size at the given position.
AIIR_CAPI_EXPORTED int64_t
aiirUniformQuantizedSubChannelTypeGetBlockSize(AiirType type, intptr_t pos);

/// Returns the scales of the quantized type.
AIIR_CAPI_EXPORTED AiirAttribute
aiirUniformQuantizedSubChannelTypeGetScales(AiirType type);

/// Returns the zero-points of the quantized type.
AIIR_CAPI_EXPORTED AiirAttribute
aiirUniformQuantizedSubChannelTypeGetZeroPoints(AiirType type);

//===---------------------------------------------------------------------===//
// CalibratedQuantizedType
//===---------------------------------------------------------------------===//

/// Returns `true` if the given type is a CalibratedQuantizedType.
AIIR_CAPI_EXPORTED bool aiirTypeIsACalibratedQuantizedType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirCalibratedQuantizedTypeGetTypeID(void);

/// Creates an instance of CalibratedQuantizedType with the given parameters
/// in the same context as `expressedType` and returns it. The instance is owned
/// by the context.
AIIR_CAPI_EXPORTED AiirType
aiirCalibratedQuantizedTypeGet(AiirType expressedType, double min, double max);

AIIR_CAPI_EXPORTED AiirStringRef aiirCalibratedQuantizedTypeGetName(void);

/// Returns the min value of the given calibrated quantized type.
AIIR_CAPI_EXPORTED double aiirCalibratedQuantizedTypeGetMin(AiirType type);

/// Returns the max value of the given calibrated quantized type.
AIIR_CAPI_EXPORTED double aiirCalibratedQuantizedTypeGetMax(AiirType type);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_QUANT_H
