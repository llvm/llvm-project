//===-- aiir-c/AffineMap.h - C API for AIIR Affine maps -----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_AFFINEMAP_H
#define AIIR_C_AFFINEMAP_H

#include "aiir-c/AffineExpr.h"
#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque type declarations.
//
// Types are exposed to C bindings as structs containing opaque pointers. They
// are not supposed to be inspected from C. This allows the underlying
// representation to change without affecting the API users. The use of structs
// instead of typedefs enables some type safety as structs are not implicitly
// convertible to each other.
//
// Instances of these types may or may not own the underlying object. The
// ownership semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirAffineMap, const void);

#undef DEFINE_C_API_STRUCT

/// Gets the context that the given affine map was created with
AIIR_CAPI_EXPORTED AiirContext aiirAffineMapGetContext(AiirAffineMap affineMap);

/// Checks whether an affine map is null.
static inline bool aiirAffineMapIsNull(AiirAffineMap affineMap) {
  return !affineMap.ptr;
}

/// Checks if two affine maps are equal.
AIIR_CAPI_EXPORTED bool aiirAffineMapEqual(AiirAffineMap a1, AiirAffineMap a2);

/// Prints an affine map by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirAffineMapPrint(AiirAffineMap affineMap,
                                           AiirStringCallback callback,
                                           void *userData);

/// Prints the affine map to the standard error stream.
AIIR_CAPI_EXPORTED void aiirAffineMapDump(AiirAffineMap affineMap);

/// Creates a zero result affine map with no dimensions or symbols in the
/// context. The affine map is owned by the context.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapEmptyGet(AiirContext ctx);

/// Creates a zero result affine map of the given dimensions and symbols in the
/// context. The affine map is owned by the context.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapZeroResultGet(
    AiirContext ctx, intptr_t dimCount, intptr_t symbolCount);

/// Creates an affine map with results defined by the given list of affine
/// expressions. The map resulting map also has the requested number of input
/// dimensions and symbols, regardless of them being used in the results.

AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapGet(AiirContext ctx,
                                                  intptr_t dimCount,
                                                  intptr_t symbolCount,
                                                  intptr_t nAffineExprs,
                                                  AiirAffineExpr *affineExprs);

/// Creates a single constant result affine map in the context. The affine map
/// is owned by the context.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapConstantGet(AiirContext ctx,
                                                          int64_t val);

/// Creates an affine map with 'numDims' identity in the context. The affine map
/// is owned by the context.
AIIR_CAPI_EXPORTED AiirAffineMap
aiirAffineMapMultiDimIdentityGet(AiirContext ctx, intptr_t numDims);

/// Creates an identity affine map on the most minor dimensions in the context.
/// The affine map is owned by the context. The function asserts that the number
/// of dimensions is greater or equal to the number of results.
AIIR_CAPI_EXPORTED AiirAffineMap
aiirAffineMapMinorIdentityGet(AiirContext ctx, intptr_t dims, intptr_t results);

/// Creates an affine map with a permutation expression and its size in the
/// context. The permutation expression is a non-empty vector of integers.
/// The elements of the permutation vector must be continuous from 0 and cannot
/// be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is
/// an invalid permutation.) The affine map is owned by the context.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapPermutationGet(
    AiirContext ctx, intptr_t size, unsigned *permutation);

/// Checks whether the given affine map is an identity affine map. The function
/// asserts that the number of dimensions is greater or equal to the number of
/// results.
AIIR_CAPI_EXPORTED bool aiirAffineMapIsIdentity(AiirAffineMap affineMap);

/// Checks whether the given affine map is a minor identity affine map.
AIIR_CAPI_EXPORTED bool aiirAffineMapIsMinorIdentity(AiirAffineMap affineMap);

/// Checks whether the given affine map is an empty affine map.
AIIR_CAPI_EXPORTED bool aiirAffineMapIsEmpty(AiirAffineMap affineMap);

/// Checks whether the given affine map is a single result constant affine
/// map.
AIIR_CAPI_EXPORTED bool aiirAffineMapIsSingleConstant(AiirAffineMap affineMap);

/// Returns the constant result of the given affine map. The function asserts
/// that the map has a single constant result.
AIIR_CAPI_EXPORTED int64_t
aiirAffineMapGetSingleConstantResult(AiirAffineMap affineMap);

/// Returns the number of dimensions of the given affine map.
AIIR_CAPI_EXPORTED intptr_t aiirAffineMapGetNumDims(AiirAffineMap affineMap);

/// Returns the number of symbols of the given affine map.
AIIR_CAPI_EXPORTED intptr_t aiirAffineMapGetNumSymbols(AiirAffineMap affineMap);

/// Returns the number of results of the given affine map.
AIIR_CAPI_EXPORTED intptr_t aiirAffineMapGetNumResults(AiirAffineMap affineMap);

/// Returns the result at the given position.
AIIR_CAPI_EXPORTED AiirAffineExpr
aiirAffineMapGetResult(AiirAffineMap affineMap, intptr_t pos);

/// Returns the number of inputs (dimensions + symbols) of the given affine
/// map.
AIIR_CAPI_EXPORTED intptr_t aiirAffineMapGetNumInputs(AiirAffineMap affineMap);

/// Checks whether the given affine map represents a subset of a symbol-less
/// permutation map.
AIIR_CAPI_EXPORTED bool
aiirAffineMapIsProjectedPermutation(AiirAffineMap affineMap);

/// Checks whether the given affine map represents a symbol-less permutation
/// map.
AIIR_CAPI_EXPORTED bool aiirAffineMapIsPermutation(AiirAffineMap affineMap);

/// Returns the affine map consisting of the `resultPos` subset.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapGetSubMap(AiirAffineMap affineMap,
                                                        intptr_t size,
                                                        intptr_t *resultPos);

/// Returns the affine map consisting of the most major `numResults` results.
/// Returns the null AffineMap if the `numResults` is equal to zero.
/// Returns the `affineMap` if `numResults` is greater or equals to number of
/// results of the given affine map.
AIIR_CAPI_EXPORTED AiirAffineMap
aiirAffineMapGetMajorSubMap(AiirAffineMap affineMap, intptr_t numResults);

/// Returns the affine map consisting of the most minor `numResults` results.
/// Returns the null AffineMap if the `numResults` is equal to zero.
/// Returns the `affineMap` if `numResults` is greater or equals to number of
/// results of the given affine map.
AIIR_CAPI_EXPORTED AiirAffineMap
aiirAffineMapGetMinorSubMap(AiirAffineMap affineMap, intptr_t numResults);

/// Apply AffineExpr::replace(`map`) to each of the results and return a new
/// new AffineMap with the new results and the specified number of dims and
/// symbols.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapReplace(
    AiirAffineMap affineMap, AiirAffineExpr expression,
    AiirAffineExpr replacement, intptr_t numResultDims, intptr_t numResultSyms);

/// Returns the simplified affine map resulting from dropping the symbols that
/// do not appear in any of the individual maps in `affineMaps`.
/// Asserts that all maps in `affineMaps` are normalized to the same number of
/// dims and symbols.
/// Takes a callback `populateResult` to fill the `res` container with value
/// `m` at entry `idx`. This allows returning without worrying about ownership
/// considerations.
AIIR_CAPI_EXPORTED void aiirAffineMapCompressUnusedSymbols(
    AiirAffineMap *affineMaps, intptr_t size, void *result,
    void (*populateResult)(void *res, intptr_t idx, AiirAffineMap m));

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_AFFINEMAP_H
