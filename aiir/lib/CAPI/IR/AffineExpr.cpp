//===- AffineExpr.cpp - C API for AIIR Affine Expressions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/AffineExpr.h"
#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"
#include "aiir/CAPI/AffineExpr.h"
#include "aiir/CAPI/AffineMap.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/IR/AffineExpr.h"

using namespace aiir;

AiirContext aiirAffineExprGetContext(AiirAffineExpr affineExpr) {
  return wrap(unwrap(affineExpr).getContext());
}

bool aiirAffineExprEqual(AiirAffineExpr lhs, AiirAffineExpr rhs) {
  return unwrap(lhs) == unwrap(rhs);
}

void aiirAffineExprPrint(AiirAffineExpr affineExpr, AiirStringCallback callback,
                         void *userData) {
  aiir::detail::CallbackOstream stream(callback, userData);
  unwrap(affineExpr).print(stream);
}

void aiirAffineExprDump(AiirAffineExpr affineExpr) {
  unwrap(affineExpr).dump();
}

bool aiirAffineExprIsSymbolicOrConstant(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).isSymbolicOrConstant();
}

bool aiirAffineExprIsPureAffine(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).isPureAffine();
}

int64_t aiirAffineExprGetLargestKnownDivisor(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).getLargestKnownDivisor();
}

bool aiirAffineExprIsMultipleOf(AiirAffineExpr affineExpr, int64_t factor) {
  return unwrap(affineExpr).isMultipleOf(factor);
}

bool aiirAffineExprIsFunctionOfDim(AiirAffineExpr affineExpr,
                                   intptr_t position) {
  return unwrap(affineExpr).isFunctionOfDim(position);
}

AiirAffineExpr aiirAffineExprCompose(AiirAffineExpr affineExpr,
                                     AiirAffineMap affineMap) {
  return wrap(unwrap(affineExpr).compose(unwrap(affineMap)));
}

AiirAffineExpr aiirAffineExprShiftDims(AiirAffineExpr affineExpr,
                                       uint32_t numDims, uint32_t shift,
                                       uint32_t offset) {
  return wrap(unwrap(affineExpr).shiftDims(numDims, shift, offset));
}

AiirAffineExpr aiirAffineExprShiftSymbols(AiirAffineExpr affineExpr,
                                          uint32_t numSymbols, uint32_t shift,
                                          uint32_t offset) {
  return wrap(unwrap(affineExpr).shiftSymbols(numSymbols, shift, offset));
}

AiirAffineExpr aiirSimplifyAffineExpr(AiirAffineExpr expr, uint32_t numDims,
                                      uint32_t numSymbols) {
  return wrap(simplifyAffineExpr(unwrap(expr), numDims, numSymbols));
}

//===----------------------------------------------------------------------===//
// Affine Dimension Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsADim(AiirAffineExpr affineExpr) {
  return isa<AffineDimExpr>(unwrap(affineExpr));
}

AiirAffineExpr aiirAffineDimExprGet(AiirContext ctx, intptr_t position) {
  return wrap(getAffineDimExpr(position, unwrap(ctx)));
}

intptr_t aiirAffineDimExprGetPosition(AiirAffineExpr affineExpr) {
  return cast<AffineDimExpr>(unwrap(affineExpr)).getPosition();
}

//===----------------------------------------------------------------------===//
// Affine Symbol Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsASymbol(AiirAffineExpr affineExpr) {
  return isa<AffineSymbolExpr>(unwrap(affineExpr));
}

AiirAffineExpr aiirAffineSymbolExprGet(AiirContext ctx, intptr_t position) {
  return wrap(getAffineSymbolExpr(position, unwrap(ctx)));
}

intptr_t aiirAffineSymbolExprGetPosition(AiirAffineExpr affineExpr) {
  return cast<AffineSymbolExpr>(unwrap(affineExpr)).getPosition();
}

//===----------------------------------------------------------------------===//
// Affine Constant Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsAConstant(AiirAffineExpr affineExpr) {
  return isa<AffineConstantExpr>(unwrap(affineExpr));
}

AiirAffineExpr aiirAffineConstantExprGet(AiirContext ctx, int64_t constant) {
  return wrap(getAffineConstantExpr(constant, unwrap(ctx)));
}

int64_t aiirAffineConstantExprGetValue(AiirAffineExpr affineExpr) {
  return cast<AffineConstantExpr>(unwrap(affineExpr)).getValue();
}

//===----------------------------------------------------------------------===//
// Affine Add Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsAAdd(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).getKind() == aiir::AffineExprKind::Add;
}

AiirAffineExpr aiirAffineAddExprGet(AiirAffineExpr lhs, AiirAffineExpr rhs) {
  return wrap(getAffineBinaryOpExpr(aiir::AffineExprKind::Add, unwrap(lhs),
                                    unwrap(rhs)));
}

//===----------------------------------------------------------------------===//
// Affine Mul Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsAMul(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).getKind() == aiir::AffineExprKind::Mul;
}

AiirAffineExpr aiirAffineMulExprGet(AiirAffineExpr lhs, AiirAffineExpr rhs) {
  return wrap(getAffineBinaryOpExpr(aiir::AffineExprKind::Mul, unwrap(lhs),
                                    unwrap(rhs)));
}

//===----------------------------------------------------------------------===//
// Affine Mod Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsAMod(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).getKind() == aiir::AffineExprKind::Mod;
}

AiirAffineExpr aiirAffineModExprGet(AiirAffineExpr lhs, AiirAffineExpr rhs) {
  return wrap(getAffineBinaryOpExpr(aiir::AffineExprKind::Mod, unwrap(lhs),
                                    unwrap(rhs)));
}

//===----------------------------------------------------------------------===//
// Affine FloorDiv Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsAFloorDiv(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).getKind() == aiir::AffineExprKind::FloorDiv;
}

AiirAffineExpr aiirAffineFloorDivExprGet(AiirAffineExpr lhs,
                                         AiirAffineExpr rhs) {
  return wrap(getAffineBinaryOpExpr(aiir::AffineExprKind::FloorDiv, unwrap(lhs),
                                    unwrap(rhs)));
}

//===----------------------------------------------------------------------===//
// Affine CeilDiv Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsACeilDiv(AiirAffineExpr affineExpr) {
  return unwrap(affineExpr).getKind() == aiir::AffineExprKind::CeilDiv;
}

AiirAffineExpr aiirAffineCeilDivExprGet(AiirAffineExpr lhs,
                                        AiirAffineExpr rhs) {
  return wrap(getAffineBinaryOpExpr(aiir::AffineExprKind::CeilDiv, unwrap(lhs),
                                    unwrap(rhs)));
}

//===----------------------------------------------------------------------===//
// Affine Binary Operation Expression.
//===----------------------------------------------------------------------===//

bool aiirAffineExprIsABinary(AiirAffineExpr affineExpr) {
  return isa<AffineBinaryOpExpr>(unwrap(affineExpr));
}

AiirAffineExpr aiirAffineBinaryOpExprGetLHS(AiirAffineExpr affineExpr) {
  return wrap(cast<AffineBinaryOpExpr>(unwrap(affineExpr)).getLHS());
}

AiirAffineExpr aiirAffineBinaryOpExprGetRHS(AiirAffineExpr affineExpr) {
  return wrap(cast<AffineBinaryOpExpr>(unwrap(affineExpr)).getRHS());
}
