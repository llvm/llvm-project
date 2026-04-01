//===- AffineMap.cpp - C API for AIIR Affine Maps -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"
#include "aiir/CAPI/AffineExpr.h"
#include "aiir/CAPI/AffineMap.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/IR/AffineMap.h"

// TODO: expose the C API related to `AffineExpr` and mutable affine map.

using namespace aiir;

AiirContext aiirAffineMapGetContext(AiirAffineMap affineMap) {
  return wrap(unwrap(affineMap).getContext());
}

bool aiirAffineMapEqual(AiirAffineMap a1, AiirAffineMap a2) {
  return unwrap(a1) == unwrap(a2);
}

void aiirAffineMapPrint(AiirAffineMap affineMap, AiirStringCallback callback,
                        void *userData) {
  aiir::detail::CallbackOstream stream(callback, userData);
  unwrap(affineMap).print(stream);
}

void aiirAffineMapDump(AiirAffineMap affineMap) { unwrap(affineMap).dump(); }

AiirAffineMap aiirAffineMapEmptyGet(AiirContext ctx) {
  return wrap(AffineMap::get(unwrap(ctx)));
}

AiirAffineMap aiirAffineMapZeroResultGet(AiirContext ctx, intptr_t dimCount,
                                         intptr_t symbolCount) {
  return wrap(AffineMap::get(dimCount, symbolCount, unwrap(ctx)));
}

AiirAffineMap aiirAffineMapGet(AiirContext ctx, intptr_t dimCount,
                               intptr_t symbolCount, intptr_t nAffineExprs,
                               AiirAffineExpr *affineExprs) {
  SmallVector<AffineExpr, 4> exprs;
  ArrayRef<AffineExpr> exprList = unwrapList(nAffineExprs, affineExprs, exprs);
  return wrap(AffineMap::get(dimCount, symbolCount, exprList, unwrap(ctx)));
}

AiirAffineMap aiirAffineMapConstantGet(AiirContext ctx, int64_t val) {
  return wrap(AffineMap::getConstantMap(val, unwrap(ctx)));
}

AiirAffineMap aiirAffineMapMultiDimIdentityGet(AiirContext ctx,
                                               intptr_t numDims) {
  return wrap(AffineMap::getMultiDimIdentityMap(numDims, unwrap(ctx)));
}

AiirAffineMap aiirAffineMapMinorIdentityGet(AiirContext ctx, intptr_t dims,
                                            intptr_t results) {
  return wrap(AffineMap::getMinorIdentityMap(dims, results, unwrap(ctx)));
}

AiirAffineMap aiirAffineMapPermutationGet(AiirContext ctx, intptr_t size,
                                          unsigned *permutation) {
  return wrap(AffineMap::getPermutationMap(
      llvm::ArrayRef(permutation, static_cast<size_t>(size)), unwrap(ctx)));
}

bool aiirAffineMapIsIdentity(AiirAffineMap affineMap) {
  return unwrap(affineMap).isIdentity();
}

bool aiirAffineMapIsMinorIdentity(AiirAffineMap affineMap) {
  return unwrap(affineMap).isMinorIdentity();
}

bool aiirAffineMapIsEmpty(AiirAffineMap affineMap) {
  return unwrap(affineMap).isEmpty();
}

bool aiirAffineMapIsSingleConstant(AiirAffineMap affineMap) {
  return unwrap(affineMap).isSingleConstant();
}

int64_t aiirAffineMapGetSingleConstantResult(AiirAffineMap affineMap) {
  return unwrap(affineMap).getSingleConstantResult();
}

intptr_t aiirAffineMapGetNumDims(AiirAffineMap affineMap) {
  return unwrap(affineMap).getNumDims();
}

intptr_t aiirAffineMapGetNumSymbols(AiirAffineMap affineMap) {
  return unwrap(affineMap).getNumSymbols();
}

intptr_t aiirAffineMapGetNumResults(AiirAffineMap affineMap) {
  return unwrap(affineMap).getNumResults();
}

AiirAffineExpr aiirAffineMapGetResult(AiirAffineMap affineMap, intptr_t pos) {
  return wrap(unwrap(affineMap).getResult(static_cast<unsigned>(pos)));
}

intptr_t aiirAffineMapGetNumInputs(AiirAffineMap affineMap) {
  return unwrap(affineMap).getNumInputs();
}

bool aiirAffineMapIsProjectedPermutation(AiirAffineMap affineMap) {
  return unwrap(affineMap).isProjectedPermutation();
}

bool aiirAffineMapIsPermutation(AiirAffineMap affineMap) {
  return unwrap(affineMap).isPermutation();
}

AiirAffineMap aiirAffineMapGetSubMap(AiirAffineMap affineMap, intptr_t size,
                                     intptr_t *resultPos) {
  SmallVector<unsigned, 8> pos;
  pos.reserve(size);
  for (intptr_t i = 0; i < size; ++i)
    pos.push_back(static_cast<unsigned>(resultPos[i]));
  return wrap(unwrap(affineMap).getSubMap(pos));
}

AiirAffineMap aiirAffineMapGetMajorSubMap(AiirAffineMap affineMap,
                                          intptr_t numResults) {
  return wrap(unwrap(affineMap).getMajorSubMap(numResults));
}

AiirAffineMap aiirAffineMapGetMinorSubMap(AiirAffineMap affineMap,
                                          intptr_t numResults) {
  return wrap(unwrap(affineMap).getMinorSubMap(numResults));
}

AiirAffineMap aiirAffineMapReplace(AiirAffineMap affineMap,
                                   AiirAffineExpr expression,
                                   AiirAffineExpr replacement,
                                   intptr_t numResultDims,
                                   intptr_t numResultSyms) {
  return wrap(unwrap(affineMap).replace(unwrap(expression), unwrap(replacement),
                                        numResultDims, numResultSyms));
}

void aiirAffineMapCompressUnusedSymbols(
    AiirAffineMap *affineMaps, intptr_t size, void *result,
    void (*populateResult)(void *res, intptr_t idx, AiirAffineMap m)) {
  SmallVector<AffineMap> maps;
  for (intptr_t idx = 0; idx < size; ++idx)
    maps.push_back(unwrap(affineMaps[idx]));
  intptr_t idx = 0;
  for (auto m : aiir::compressUnusedSymbols(maps))
    populateResult(result, idx++, wrap(m));
}
