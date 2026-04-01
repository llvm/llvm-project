//===- IntegerSet.cpp - C API for AIIR Integer Sets -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/IntegerSet.h"
#include "aiir-c/AffineExpr.h"
#include "aiir/CAPI/AffineExpr.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/IntegerSet.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/IR/IntegerSet.h"

using namespace aiir;

AiirContext aiirIntegerSetGetContext(AiirIntegerSet set) {
  return wrap(unwrap(set).getContext());
}

bool aiirIntegerSetEqual(AiirIntegerSet s1, AiirIntegerSet s2) {
  return unwrap(s1) == unwrap(s2);
}

void aiirIntegerSetPrint(AiirIntegerSet set, AiirStringCallback callback,
                         void *userData) {
  aiir::detail::CallbackOstream stream(callback, userData);
  unwrap(set).print(stream);
}

void aiirIntegerSetDump(AiirIntegerSet set) { unwrap(set).dump(); }

AiirIntegerSet aiirIntegerSetEmptyGet(AiirContext context, intptr_t numDims,
                                      intptr_t numSymbols) {
  return wrap(IntegerSet::getEmptySet(static_cast<unsigned>(numDims),
                                      static_cast<unsigned>(numSymbols),
                                      unwrap(context)));
}

AiirIntegerSet aiirIntegerSetGet(AiirContext context, intptr_t numDims,
                                 intptr_t numSymbols, intptr_t numConstraints,
                                 const AiirAffineExpr *constraints,
                                 const bool *eqFlags) {
  SmallVector<AffineExpr> aiirConstraints;
  (void)unwrapList(static_cast<size_t>(numConstraints), constraints,
                   aiirConstraints);
  return wrap(IntegerSet::get(
      static_cast<unsigned>(numDims), static_cast<unsigned>(numSymbols),
      aiirConstraints,
      llvm::ArrayRef(eqFlags, static_cast<size_t>(numConstraints))));
}

AiirIntegerSet
aiirIntegerSetReplaceGet(AiirIntegerSet set,
                         const AiirAffineExpr *dimReplacements,
                         const AiirAffineExpr *symbolReplacements,
                         intptr_t numResultDims, intptr_t numResultSymbols) {
  SmallVector<AffineExpr> aiirDims, aiirSymbols;
  (void)unwrapList(unwrap(set).getNumDims(), dimReplacements, aiirDims);
  (void)unwrapList(unwrap(set).getNumSymbols(), symbolReplacements,
                   aiirSymbols);
  return wrap(unwrap(set).replaceDimsAndSymbols(
      aiirDims, aiirSymbols, static_cast<unsigned>(numResultDims),
      static_cast<unsigned>(numResultSymbols)));
}

bool aiirIntegerSetIsCanonicalEmpty(AiirIntegerSet set) {
  return unwrap(set).isEmptyIntegerSet();
}

intptr_t aiirIntegerSetGetNumDims(AiirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumDims());
}

intptr_t aiirIntegerSetGetNumSymbols(AiirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumSymbols());
}

intptr_t aiirIntegerSetGetNumInputs(AiirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumInputs());
}

intptr_t aiirIntegerSetGetNumConstraints(AiirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumConstraints());
}

intptr_t aiirIntegerSetGetNumEqualities(AiirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumEqualities());
}

intptr_t aiirIntegerSetGetNumInequalities(AiirIntegerSet set) {
  return static_cast<intptr_t>(unwrap(set).getNumInequalities());
}

AiirAffineExpr aiirIntegerSetGetConstraint(AiirIntegerSet set, intptr_t pos) {
  return wrap(unwrap(set).getConstraint(static_cast<unsigned>(pos)));
}

bool aiirIntegerSetIsConstraintEq(AiirIntegerSet set, intptr_t pos) {
  return unwrap(set).isEq(pos);
}
