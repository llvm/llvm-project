//===-- aiir-c/BuiltinTypes.h - C API for AIIR Builtin types ------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_BUILTINTYPES_H
#define AIIR_C_BUILTINTYPES_H

#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Integer types.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Integer type.
AIIR_CAPI_EXPORTED AiirTypeID aiirIntegerTypeGetTypeID(void);

/// Checks whether the given type is an integer type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAInteger(AiirType type);

/// Creates a signless integer type of the given bitwidth in the context. The
/// type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirIntegerTypeGet(AiirContext ctx,
                                               unsigned bitwidth);

AIIR_CAPI_EXPORTED AiirStringRef aiirIntegerTypeGetName(void);

/// Creates a signed integer type of the given bitwidth in the context. The type
/// is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirIntegerTypeSignedGet(AiirContext ctx,
                                                     unsigned bitwidth);

/// Creates an unsigned integer type of the given bitwidth in the context. The
/// type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirIntegerTypeUnsignedGet(AiirContext ctx,
                                                       unsigned bitwidth);

/// Returns the bitwidth of an integer type.
AIIR_CAPI_EXPORTED unsigned aiirIntegerTypeGetWidth(AiirType type);

/// Checks whether the given integer type is signless.
AIIR_CAPI_EXPORTED bool aiirIntegerTypeIsSignless(AiirType type);

/// Checks whether the given integer type is signed.
AIIR_CAPI_EXPORTED bool aiirIntegerTypeIsSigned(AiirType type);

/// Checks whether the given integer type is unsigned.
AIIR_CAPI_EXPORTED bool aiirIntegerTypeIsUnsigned(AiirType type);

//===----------------------------------------------------------------------===//
// Index type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Index type.
AIIR_CAPI_EXPORTED AiirTypeID aiirIndexTypeGetTypeID(void);

/// Checks whether the given type is an index type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAIndex(AiirType type);

/// Creates an index type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirIndexTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirIndexTypeGetName(void);

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a floating-point type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat(AiirType type);

/// Returns the bitwidth of a floating-point type.
AIIR_CAPI_EXPORTED unsigned aiirFloatTypeGetWidth(AiirType type);

/// Returns the typeID of an Float4E2M1FN type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat4E2M1FNTypeGetTypeID(void);

/// Checks whether the given type is an f4E2M1FN type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat4E2M1FN(AiirType type);

/// Creates an f4E2M1FN type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat4E2M1FNTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat4E2M1FNTypeGetName(void);

/// Returns the typeID of an Float6E2M3FN type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat6E2M3FNTypeGetTypeID(void);

/// Checks whether the given type is an f6E2M3FN type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat6E2M3FN(AiirType type);

/// Creates an f6E2M3FN type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat6E2M3FNTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat6E2M3FNTypeGetName(void);

/// Returns the typeID of an Float6E3M2FN type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat6E3M2FNTypeGetTypeID(void);

/// Checks whether the given type is an f6E3M2FN type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat6E3M2FN(AiirType type);

/// Creates an f6E3M2FN type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat6E3M2FNTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat6E3M2FNTypeGetName(void);

/// Returns the typeID of an Float8E5M2 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E5M2TypeGetTypeID(void);

/// Checks whether the given type is an f8E5M2 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E5M2(AiirType type);

/// Creates an f8E5M2 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E5M2TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E5M2TypeGetName(void);

/// Returns the typeID of an Float8E4M3 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E4M3TypeGetTypeID(void);

/// Checks whether the given type is an f8E4M3 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E4M3(AiirType type);

/// Creates an f8E4M3 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E4M3TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E4M3TypeGetName(void);

/// Returns the typeID of an Float8E4M3FN type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E4M3FNTypeGetTypeID(void);

/// Checks whether the given type is an f8E4M3FN type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E4M3FN(AiirType type);

/// Creates an f8E4M3FN type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E4M3FNTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E4M3FNTypeGetName(void);

/// Returns the typeID of an Float8E5M2FNUZ type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E5M2FNUZTypeGetTypeID(void);

/// Checks whether the given type is an f8E5M2FNUZ type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E5M2FNUZ(AiirType type);

/// Creates an f8E5M2FNUZ type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E5M2FNUZTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E5M2FNUZTypeGetName(void);

/// Returns the typeID of an Float8E4M3FNUZ type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E4M3FNUZTypeGetTypeID(void);

/// Checks whether the given type is an f8E4M3FNUZ type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E4M3FNUZ(AiirType type);

/// Creates an f8E4M3FNUZ type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E4M3FNUZTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E4M3FNUZTypeGetName(void);

/// Returns the typeID of an Float8E4M3B11FNUZ type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E4M3B11FNUZTypeGetTypeID(void);

/// Checks whether the given type is an f8E4M3B11FNUZ type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E4M3B11FNUZ(AiirType type);

/// Creates an f8E4M3B11FNUZ type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E4M3B11FNUZTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E4M3B11FNUZTypeGetName(void);

/// Returns the typeID of an Float8E3M4 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E3M4TypeGetTypeID(void);

/// Checks whether the given type is an f8E3M4 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E3M4(AiirType type);

/// Creates an f8E3M4 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E3M4TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E3M4TypeGetName(void);

/// Returns the typeID of an Float8E8M0FNU type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat8E8M0FNUTypeGetTypeID(void);

/// Checks whether the given type is an f8E8M0FNU type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFloat8E8M0FNU(AiirType type);

/// Creates an f8E8M0FNU type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirFloat8E8M0FNUTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloat8E8M0FNUTypeGetName(void);

/// Returns the typeID of an BFloat16 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirBFloat16TypeGetTypeID(void);

/// Checks whether the given type is a bf16 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsABF16(AiirType type);

/// Creates a bf16 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirBF16TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirBF16TypeGetName(void);

/// Returns the typeID of an Float16 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat16TypeGetTypeID(void);

/// Checks whether the given type is an f16 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAF16(AiirType type);

/// Creates an f16 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirF16TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirF16TypeGetName(void);

/// Returns the typeID of an Float32 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat32TypeGetTypeID(void);

/// Checks whether the given type is an f32 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAF32(AiirType type);

/// Creates an f32 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirF32TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirF32TypeGetName(void);

/// Returns the typeID of an Float64 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloat64TypeGetTypeID(void);

/// Checks whether the given type is an f64 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAF64(AiirType type);

/// Creates a f64 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirF64TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirF64TypeGetName(void);

/// Returns the typeID of a TF32 type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloatTF32TypeGetTypeID(void);

/// Checks whether the given type is an TF32 type.
AIIR_CAPI_EXPORTED bool aiirTypeIsATF32(AiirType type);

/// Creates a TF32 type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirTF32TypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirTF32TypeGetName(void);

//===----------------------------------------------------------------------===//
// None type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an None type.
AIIR_CAPI_EXPORTED AiirTypeID aiirNoneTypeGetTypeID(void);

/// Checks whether the given type is a None type.
AIIR_CAPI_EXPORTED bool aiirTypeIsANone(AiirType type);

/// Creates a None type in the given context. The type is owned by the
/// context.
AIIR_CAPI_EXPORTED AiirType aiirNoneTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirNoneTypeGetName(void);

//===----------------------------------------------------------------------===//
// Complex type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Complex type.
AIIR_CAPI_EXPORTED AiirTypeID aiirComplexTypeGetTypeID(void);

/// Checks whether the given type is a Complex type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAComplex(AiirType type);

/// Creates a complex type with the given element type in the same context as
/// the element type. The type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirComplexTypeGet(AiirType elementType);

AIIR_CAPI_EXPORTED AiirStringRef aiirComplexTypeGetName(void);

/// Returns the element type of the given complex type.
AIIR_CAPI_EXPORTED AiirType aiirComplexTypeGetElementType(AiirType type);

//===----------------------------------------------------------------------===//
// Shaped type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a Shaped type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAShaped(AiirType type);

/// Returns the element type of the shaped type.
AIIR_CAPI_EXPORTED AiirType aiirShapedTypeGetElementType(AiirType type);

/// Checks whether the given shaped type is ranked.
AIIR_CAPI_EXPORTED bool aiirShapedTypeHasRank(AiirType type);

/// Returns the rank of the given ranked shaped type.
AIIR_CAPI_EXPORTED int64_t aiirShapedTypeGetRank(AiirType type);

/// Checks whether the given shaped type has a static shape.
AIIR_CAPI_EXPORTED bool aiirShapedTypeHasStaticShape(AiirType type);

/// Checks whether the dim-th dimension of the given shaped type is dynamic.
AIIR_CAPI_EXPORTED bool aiirShapedTypeIsDynamicDim(AiirType type, intptr_t dim);

/// Checks whether the dim-th dimension of the given shaped type is static.
AIIR_CAPI_EXPORTED bool aiirShapedTypeIsStaticDim(AiirType type, intptr_t dim);

/// Returns the dim-th dimension of the given ranked shaped type.
AIIR_CAPI_EXPORTED int64_t aiirShapedTypeGetDimSize(AiirType type,
                                                    intptr_t dim);

/// Checks whether the given value is used as a placeholder for dynamic sizes
/// in shaped types.
AIIR_CAPI_EXPORTED bool aiirShapedTypeIsDynamicSize(int64_t size);

/// Checks whether the given shaped type dimension value is statically-sized.
AIIR_CAPI_EXPORTED bool aiirShapedTypeIsStaticSize(int64_t size);

/// Returns the value indicating a dynamic size in a shaped type. Prefer
/// aiirShapedTypeIsDynamicSize and aiirShapedTypeIsStaticSize to direct
/// comparisons with this value.
AIIR_CAPI_EXPORTED int64_t aiirShapedTypeGetDynamicSize(void);

/// Checks whether the given value is used as a placeholder for dynamic strides
/// and offsets in shaped types.
AIIR_CAPI_EXPORTED bool aiirShapedTypeIsDynamicStrideOrOffset(int64_t val);

/// Checks whether the given dimension value of a stride or an offset is
/// statically-sized.
AIIR_CAPI_EXPORTED bool aiirShapedTypeIsStaticStrideOrOffset(int64_t val);

/// Returns the value indicating a dynamic stride or offset in a shaped type.
/// Prefer aiirShapedTypeIsDynamicStrideOrOffset and
/// aiirShapedTypeIsStaticStrideOrOffset to direct comparisons with this value.
AIIR_CAPI_EXPORTED int64_t aiirShapedTypeGetDynamicStrideOrOffset(void);

//===----------------------------------------------------------------------===//
// Vector type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Vector type.
AIIR_CAPI_EXPORTED AiirTypeID aiirVectorTypeGetTypeID(void);

/// Checks whether the given type is a Vector type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAVector(AiirType type);

/// Creates a vector type of the shape identified by its rank and dimensions,
/// with the given element type in the same context as the element type. The
/// type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirVectorTypeGet(intptr_t rank,
                                              const int64_t *shape,
                                              AiirType elementType);

AIIR_CAPI_EXPORTED AiirStringRef aiirVectorTypeGetName(void);

/// Same as "aiirVectorTypeGet" but returns a nullptr wrapping AiirType on
/// illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED AiirType aiirVectorTypeGetChecked(AiirLocation loc,
                                                     intptr_t rank,
                                                     const int64_t *shape,
                                                     AiirType elementType);

/// Creates a scalable vector type with the shape identified by its rank and
/// dimensions. A subset of dimensions may be marked as scalable via the
/// corresponding flag list, which is expected to have as many entries as the
/// rank of the vector. The vector is created in the same context as the element
/// type.
AIIR_CAPI_EXPORTED AiirType aiirVectorTypeGetScalable(intptr_t rank,
                                                      const int64_t *shape,
                                                      const bool *scalable,
                                                      AiirType elementType);

/// Same as "aiirVectorTypeGetScalable" but returns a nullptr wrapping AiirType
/// on illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED
AiirType aiirVectorTypeGetScalableChecked(AiirLocation loc, intptr_t rank,
                                          const int64_t *shape,
                                          const bool *scalable,
                                          AiirType elementType);

/// Checks whether the given vector type is scalable, i.e., has at least one
/// scalable dimension.
AIIR_CAPI_EXPORTED bool aiirVectorTypeIsScalable(AiirType type);

/// Checks whether the "dim"-th dimension of the given vector is scalable.
AIIR_CAPI_EXPORTED bool aiirVectorTypeIsDimScalable(AiirType type,
                                                    intptr_t dim);

//===----------------------------------------------------------------------===//
// Ranked / Unranked Tensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a Tensor type.
AIIR_CAPI_EXPORTED bool aiirTypeIsATensor(AiirType type);

/// Returns the typeID of an RankedTensor type.
AIIR_CAPI_EXPORTED AiirTypeID aiirRankedTensorTypeGetTypeID(void);

/// Checks whether the given type is a ranked tensor type.
AIIR_CAPI_EXPORTED bool aiirTypeIsARankedTensor(AiirType type);

/// Returns the typeID of an UnrankedTensor type.
AIIR_CAPI_EXPORTED AiirTypeID aiirUnrankedTensorTypeGetTypeID(void);

/// Checks whether the given type is an unranked tensor type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAUnrankedTensor(AiirType type);

/// Creates a tensor type of a fixed rank with the given shape, element type,
/// and optional encoding in the same context as the element type. The type is
/// owned by the context. Tensor types without any specific encoding field
/// should assign aiirAttributeGetNull() to this parameter.
AIIR_CAPI_EXPORTED AiirType aiirRankedTensorTypeGet(intptr_t rank,
                                                    const int64_t *shape,
                                                    AiirType elementType,
                                                    AiirAttribute encoding);

AIIR_CAPI_EXPORTED AiirStringRef aiirRankedTensorTypeGetName(void);

/// Same as "aiirRankedTensorTypeGet" but returns a nullptr wrapping AiirType on
/// illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED AiirType aiirRankedTensorTypeGetChecked(
    AiirLocation loc, intptr_t rank, const int64_t *shape, AiirType elementType,
    AiirAttribute encoding);

/// Gets the 'encoding' attribute from the ranked tensor type, returning a null
/// attribute if none.
AIIR_CAPI_EXPORTED AiirAttribute aiirRankedTensorTypeGetEncoding(AiirType type);

/// Creates an unranked tensor type with the given element type in the same
/// context as the element type. The type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirUnrankedTensorTypeGet(AiirType elementType);

AIIR_CAPI_EXPORTED AiirStringRef aiirUnrankedTensorTypeGetName(void);

/// Same as "aiirUnrankedTensorTypeGet" but returns a nullptr wrapping AiirType
/// on illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED AiirType
aiirUnrankedTensorTypeGetChecked(AiirLocation loc, AiirType elementType);

//===----------------------------------------------------------------------===//
// Ranked / Unranked MemRef type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an MemRef type.
AIIR_CAPI_EXPORTED AiirTypeID aiirMemRefTypeGetTypeID(void);

/// Checks whether the given type is a MemRef type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAMemRef(AiirType type);

/// Returns the typeID of an UnrankedMemRef type.
AIIR_CAPI_EXPORTED AiirTypeID aiirUnrankedMemRefTypeGetTypeID(void);

/// Checks whether the given type is an UnrankedMemRef type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAUnrankedMemRef(AiirType type);

/// Creates a MemRef type with the given rank and shape, a potentially empty
/// list of affine layout maps, the given memory space and element type, in the
/// same context as element type. The type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirMemRefTypeGet(AiirType elementType,
                                              intptr_t rank,
                                              const int64_t *shape,
                                              AiirAttribute layout,
                                              AiirAttribute memorySpace);

AIIR_CAPI_EXPORTED AiirStringRef aiirMemRefTypeGetName(void);

/// Same as "aiirMemRefTypeGet" but returns a nullptr-wrapping AiirType o
/// illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED AiirType aiirMemRefTypeGetChecked(
    AiirLocation loc, AiirType elementType, intptr_t rank, const int64_t *shape,
    AiirAttribute layout, AiirAttribute memorySpace);

/// Creates a MemRef type with the given rank, shape, memory space and element
/// type in the same context as the element type. The type has no affine maps,
/// i.e. represents a default row-major contiguous memref. The type is owned by
/// the context.
AIIR_CAPI_EXPORTED AiirType
aiirMemRefTypeContiguousGet(AiirType elementType, intptr_t rank,
                            const int64_t *shape, AiirAttribute memorySpace);

/// Same as "aiirMemRefTypeContiguousGet" but returns a nullptr wrapping
/// AiirType on illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED AiirType aiirMemRefTypeContiguousGetChecked(
    AiirLocation loc, AiirType elementType, intptr_t rank, const int64_t *shape,
    AiirAttribute memorySpace);

/// Creates an Unranked MemRef type with the given element type and in the given
/// memory space. The type is owned by the context of element type.
AIIR_CAPI_EXPORTED AiirType
aiirUnrankedMemRefTypeGet(AiirType elementType, AiirAttribute memorySpace);

AIIR_CAPI_EXPORTED AiirStringRef aiirUnrankedMemRefTypeGetName(void);

/// Same as "aiirUnrankedMemRefTypeGet" but returns a nullptr wrapping
/// AiirType on illegal arguments, emitting appropriate diagnostics.
AIIR_CAPI_EXPORTED AiirType aiirUnrankedMemRefTypeGetChecked(
    AiirLocation loc, AiirType elementType, AiirAttribute memorySpace);

/// Returns the layout of the given MemRef type.
AIIR_CAPI_EXPORTED AiirAttribute aiirMemRefTypeGetLayout(AiirType type);

/// Returns the affine map of the given MemRef type.
AIIR_CAPI_EXPORTED AiirAffineMap aiirMemRefTypeGetAffineMap(AiirType type);

/// Returns the memory space of the given MemRef type.
AIIR_CAPI_EXPORTED AiirAttribute aiirMemRefTypeGetMemorySpace(AiirType type);

/// Returns the strides of the MemRef if the layout map is in strided form.
/// Both strides and offset are out params. strides must point to pre-allocated
/// memory of length equal to the rank of the memref.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirMemRefTypeGetStridesAndOffset(
    AiirType type, int64_t *strides, int64_t *offset);

/// Returns the memory spcae of the given Unranked MemRef type.
AIIR_CAPI_EXPORTED AiirAttribute
aiirUnrankedMemrefGetMemorySpace(AiirType type);

//===----------------------------------------------------------------------===//
// Tuple type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Tuple type.
AIIR_CAPI_EXPORTED AiirTypeID aiirTupleTypeGetTypeID(void);

/// Checks whether the given type is a tuple type.
AIIR_CAPI_EXPORTED bool aiirTypeIsATuple(AiirType type);

/// Creates a tuple type that consists of the given list of elemental types. The
/// type is owned by the context.
AIIR_CAPI_EXPORTED AiirType aiirTupleTypeGet(AiirContext ctx,
                                             intptr_t numElements,
                                             AiirType const *elements);

AIIR_CAPI_EXPORTED AiirStringRef aiirTupleTypeGetName(void);

/// Returns the number of types contained in a tuple.
AIIR_CAPI_EXPORTED intptr_t aiirTupleTypeGetNumTypes(AiirType type);

/// Returns the pos-th type in the tuple type.
AIIR_CAPI_EXPORTED AiirType aiirTupleTypeGetType(AiirType type, intptr_t pos);

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Function type.
AIIR_CAPI_EXPORTED AiirTypeID aiirFunctionTypeGetTypeID(void);

/// Checks whether the given type is a function type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAFunction(AiirType type);

/// Creates a function type, mapping a list of input types to result types.
AIIR_CAPI_EXPORTED AiirType aiirFunctionTypeGet(AiirContext ctx,
                                                intptr_t numInputs,
                                                AiirType const *inputs,
                                                intptr_t numResults,
                                                AiirType const *results);

AIIR_CAPI_EXPORTED AiirStringRef aiirFunctionTypeGetName(void);

/// Returns the number of input types.
AIIR_CAPI_EXPORTED intptr_t aiirFunctionTypeGetNumInputs(AiirType type);

/// Returns the number of result types.
AIIR_CAPI_EXPORTED intptr_t aiirFunctionTypeGetNumResults(AiirType type);

/// Returns the pos-th input type.
AIIR_CAPI_EXPORTED AiirType aiirFunctionTypeGetInput(AiirType type,
                                                     intptr_t pos);

/// Returns the pos-th result type.
AIIR_CAPI_EXPORTED AiirType aiirFunctionTypeGetResult(AiirType type,
                                                      intptr_t pos);

//===----------------------------------------------------------------------===//
// Opaque type.
//===----------------------------------------------------------------------===//

/// Returns the typeID of an Opaque type.
AIIR_CAPI_EXPORTED AiirTypeID aiirOpaqueTypeGetTypeID(void);

/// Checks whether the given type is an opaque type.
AIIR_CAPI_EXPORTED bool aiirTypeIsAOpaque(AiirType type);

/// Creates an opaque type in the given context associated with the dialect
/// identified by its namespace. The type contains opaque byte data of the
/// specified length (data need not be null-terminated).
AIIR_CAPI_EXPORTED AiirType aiirOpaqueTypeGet(AiirContext ctx,
                                              AiirStringRef dialectNamespace,
                                              AiirStringRef typeData);

AIIR_CAPI_EXPORTED AiirStringRef aiirOpaqueTypeGetName(void);

/// Returns the namespace of the dialect with which the given opaque type
/// is associated. The namespace string is owned by the context.
AIIR_CAPI_EXPORTED AiirStringRef
aiirOpaqueTypeGetDialectNamespace(AiirType type);

/// Returns the raw data as a string reference. The data remains live as long as
/// the context in which the type lives.
AIIR_CAPI_EXPORTED AiirStringRef aiirOpaqueTypeGetData(AiirType type);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_BUILTINTYPES_H
