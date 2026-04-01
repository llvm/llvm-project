//===-- aiir-c/AffineExpr.h - C API for AIIR Affine Expressions ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_AFFINEEXPR_H
#define AIIR_C_AFFINEEXPR_H

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

DEFINE_C_API_STRUCT(AiirAffineExpr, const void);

#undef DEFINE_C_API_STRUCT

struct AiirAffineMap;

/// Gets the context that owns the affine expression.
AIIR_CAPI_EXPORTED AiirContext
aiirAffineExprGetContext(AiirAffineExpr affineExpr);

/// Returns `true` if the two affine expressions are equal.
AIIR_CAPI_EXPORTED bool aiirAffineExprEqual(AiirAffineExpr lhs,
                                            AiirAffineExpr rhs);

/// Returns `true` if the given affine expression is a null expression. Note
/// constant zero is not a null expression.
inline static bool aiirAffineExprIsNull(AiirAffineExpr affineExpr) {
  return affineExpr.ptr == NULL;
}

/// Prints an affine expression by sending chunks of the string representation
/// and forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirAffineExprPrint(AiirAffineExpr affineExpr,
                                            AiirStringCallback callback,
                                            void *userData);

/// Prints the affine expression to the standard error stream.
AIIR_CAPI_EXPORTED void aiirAffineExprDump(AiirAffineExpr affineExpr);

/// Checks whether the given affine expression is made out of only symbols and
/// constants.
AIIR_CAPI_EXPORTED bool
aiirAffineExprIsSymbolicOrConstant(AiirAffineExpr affineExpr);

/// Checks whether the given affine expression is a pure affine expression, i.e.
/// mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsPureAffine(AiirAffineExpr affineExpr);

/// Returns the greatest known integral divisor of this affine expression. The
/// result is always positive.
AIIR_CAPI_EXPORTED int64_t
aiirAffineExprGetLargestKnownDivisor(AiirAffineExpr affineExpr);

/// Checks whether the given affine expression is a multiple of 'factor'.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsMultipleOf(AiirAffineExpr affineExpr,
                                                   int64_t factor);

/// Checks whether the given affine expression involves AffineDimExpr
/// 'position'.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsFunctionOfDim(AiirAffineExpr affineExpr,
                                                      intptr_t position);

/// Composes the given map with the given expression.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineExprCompose(
    AiirAffineExpr affineExpr, struct AiirAffineMap affineMap);

/// Replace dims[offset ... numDims)
/// by dims[offset + shift ... shift + numDims).
AIIR_CAPI_EXPORTED AiirAffineExpr
aiirAffineExprShiftDims(AiirAffineExpr affineExpr, uint32_t numDims,
                        uint32_t shift, uint32_t offset);

/// Replace symbols[offset ... numSymbols)
/// by symbols[offset + shift ... shift + numSymbols).
AIIR_CAPI_EXPORTED AiirAffineExpr
aiirAffineExprShiftSymbols(AiirAffineExpr affineExpr, uint32_t numSymbols,
                           uint32_t shift, uint32_t offset);

/// Simplify an affine expression by flattening and some amount of simple
/// analysis. This has complexity linear in the number of nodes in 'expr'.
/// Returns the simplified expression, which is the same as the input expression
/// if it can't be simplified. When `expr` is semi-affine, a simplified
/// semi-affine expression is constructed in the sorted order of dimension and
/// symbol positions.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirSimplifyAffineExpr(AiirAffineExpr expr,
                                                         uint32_t numDims,
                                                         uint32_t numSymbols);

//===----------------------------------------------------------------------===//
// Affine Dimension Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is a dimension expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsADim(AiirAffineExpr affineExpr);

/// Creates an affine dimension expression with 'position' in the context.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineDimExprGet(AiirContext ctx,
                                                       intptr_t position);

/// Returns the position of the given affine dimension expression.
AIIR_CAPI_EXPORTED intptr_t
aiirAffineDimExprGetPosition(AiirAffineExpr affineExpr);

//===----------------------------------------------------------------------===//
// Affine Symbol Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is a symbol expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsASymbol(AiirAffineExpr affineExpr);

/// Creates an affine symbol expression with 'position' in the context.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineSymbolExprGet(AiirContext ctx,
                                                          intptr_t position);

/// Returns the position of the given affine symbol expression.
AIIR_CAPI_EXPORTED intptr_t
aiirAffineSymbolExprGetPosition(AiirAffineExpr affineExpr);

//===----------------------------------------------------------------------===//
// Affine Constant Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is a constant expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsAConstant(AiirAffineExpr affineExpr);

/// Creates an affine constant expression with 'constant' in the context.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineConstantExprGet(AiirContext ctx,
                                                            int64_t constant);

/// Returns the value of the given affine constant expression.
AIIR_CAPI_EXPORTED int64_t
aiirAffineConstantExprGetValue(AiirAffineExpr affineExpr);

//===----------------------------------------------------------------------===//
// Affine Add Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an add expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsAAdd(AiirAffineExpr affineExpr);

/// Creates an affine add expression with 'lhs' and 'rhs'.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineAddExprGet(AiirAffineExpr lhs,
                                                       AiirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine Mul Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an mul expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsAMul(AiirAffineExpr affineExpr);

/// Creates an affine mul expression with 'lhs' and 'rhs'.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineMulExprGet(AiirAffineExpr lhs,
                                                       AiirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine Mod Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an mod expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsAMod(AiirAffineExpr affineExpr);

/// Creates an affine mod expression with 'lhs' and 'rhs'.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineModExprGet(AiirAffineExpr lhs,
                                                       AiirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine FloorDiv Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an floordiv expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsAFloorDiv(AiirAffineExpr affineExpr);

/// Creates an affine floordiv expression with 'lhs' and 'rhs'.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineFloorDivExprGet(AiirAffineExpr lhs,
                                                            AiirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine CeilDiv Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is an ceildiv expression.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsACeilDiv(AiirAffineExpr affineExpr);

/// Creates an affine ceildiv expression with 'lhs' and 'rhs'.
AIIR_CAPI_EXPORTED AiirAffineExpr aiirAffineCeilDivExprGet(AiirAffineExpr lhs,
                                                           AiirAffineExpr rhs);

//===----------------------------------------------------------------------===//
// Affine Binary Operation Expression.
//===----------------------------------------------------------------------===//

/// Checks whether the given affine expression is binary.
AIIR_CAPI_EXPORTED bool aiirAffineExprIsABinary(AiirAffineExpr affineExpr);

/// Returns the left hand side affine expression of the given affine binary
/// operation expression.
AIIR_CAPI_EXPORTED AiirAffineExpr
aiirAffineBinaryOpExprGetLHS(AiirAffineExpr affineExpr);

/// Returns the right hand side affine expression of the given affine binary
/// operation expression.
AIIR_CAPI_EXPORTED AiirAffineExpr
aiirAffineBinaryOpExprGetRHS(AiirAffineExpr affineExpr);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_AFFINEEXPR_H
