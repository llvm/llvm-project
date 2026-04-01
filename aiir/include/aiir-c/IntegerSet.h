//===-- aiir-c/IntegerSet.h - C API for AIIR Affine maps ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_INTEGERSET_H
#define AIIR_C_INTEGERSET_H

#include "aiir-c/AffineExpr.h"

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

DEFINE_C_API_STRUCT(AiirIntegerSet, const void);

#undef DEFINE_C_API_STRUCT

/// Gets the context in which the given integer set lives.
AIIR_CAPI_EXPORTED AiirContext aiirIntegerSetGetContext(AiirIntegerSet set);

/// Checks whether an integer set is a null object.
static inline bool aiirIntegerSetIsNull(AiirIntegerSet set) { return !set.ptr; }

/// Checks if two integer set objects are equal. This is a "shallow" comparison
/// of two objects. Only the sets with some small number of constraints are
/// uniqued and compare equal here. Set objects that represent the same integer
/// set with different constraints may be considered non-equal by this check.
/// Set difference followed by an (expensive) emptiness check should be used to
/// check equivalence of the underlying integer sets.
AIIR_CAPI_EXPORTED bool aiirIntegerSetEqual(AiirIntegerSet s1,
                                            AiirIntegerSet s2);

/// Prints an integer set by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
AIIR_CAPI_EXPORTED void aiirIntegerSetPrint(AiirIntegerSet set,
                                            AiirStringCallback callback,
                                            void *userData);

/// Prints an integer set to the standard error stream.
AIIR_CAPI_EXPORTED void aiirIntegerSetDump(AiirIntegerSet set);

/// Gets or creates a new canonically empty integer set with the give number of
/// dimensions and symbols in the given context.
AIIR_CAPI_EXPORTED AiirIntegerSet aiirIntegerSetEmptyGet(AiirContext context,
                                                         intptr_t numDims,
                                                         intptr_t numSymbols);

/// Gets or creates a new integer set in the given context. The set is defined
/// by a list of affine constraints, with the given number of input dimensions
/// and symbols, which are treated as either equalities (eqFlags is 1) or
/// inequalities (eqFlags is 0). Both `constraints` and `eqFlags` are expected
/// to point to at least `numConstraint` consecutive values.
AIIR_CAPI_EXPORTED AiirIntegerSet
aiirIntegerSetGet(AiirContext context, intptr_t numDims, intptr_t numSymbols,
                  intptr_t numConstraints, const AiirAffineExpr *constraints,
                  const bool *eqFlags);

/// Gets or creates a new integer set in which the values and dimensions of the
/// given set are replaced with the given affine expressions. `dimReplacements`
/// and `symbolReplacements` are expected to point to at least as many
/// consecutive expressions as the given set has dimensions and symbols,
/// respectively. The new set will have `numResultDims` and `numResultSymbols`
/// dimensions and symbols, respectively.
AIIR_CAPI_EXPORTED AiirIntegerSet aiirIntegerSetReplaceGet(
    AiirIntegerSet set, const AiirAffineExpr *dimReplacements,
    const AiirAffineExpr *symbolReplacements, intptr_t numResultDims,
    intptr_t numResultSymbols);

/// Checks whether the given set is a canonical empty set, e.g., the set
/// returned by aiirIntegerSetEmptyGet.
AIIR_CAPI_EXPORTED bool aiirIntegerSetIsCanonicalEmpty(AiirIntegerSet set);

/// Returns the number of dimensions in the given set.
AIIR_CAPI_EXPORTED intptr_t aiirIntegerSetGetNumDims(AiirIntegerSet set);

/// Returns the number of symbols in the given set.
AIIR_CAPI_EXPORTED intptr_t aiirIntegerSetGetNumSymbols(AiirIntegerSet set);

/// Returns the number of inputs (dimensions + symbols) in the given set.
AIIR_CAPI_EXPORTED intptr_t aiirIntegerSetGetNumInputs(AiirIntegerSet set);

/// Returns the number of constraints (equalities + inequalities) in the given
/// set.
AIIR_CAPI_EXPORTED intptr_t aiirIntegerSetGetNumConstraints(AiirIntegerSet set);

/// Returns the number of equalities in the given set.
AIIR_CAPI_EXPORTED intptr_t aiirIntegerSetGetNumEqualities(AiirIntegerSet set);

/// Returns the number of inequalities in the given set.
AIIR_CAPI_EXPORTED intptr_t
aiirIntegerSetGetNumInequalities(AiirIntegerSet set);

/// Returns `pos`-th constraint of the set.
AIIR_CAPI_EXPORTED AiirAffineExpr
aiirIntegerSetGetConstraint(AiirIntegerSet set, intptr_t pos);

/// Returns `true` of the `pos`-th constraint of the set is an equality
/// constraint, `false` otherwise.
AIIR_CAPI_EXPORTED bool aiirIntegerSetIsConstraintEq(AiirIntegerSet set,
                                                     intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_INTEGERSET_H
