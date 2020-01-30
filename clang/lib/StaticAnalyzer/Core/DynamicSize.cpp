//===- DynamicSize.cpp - Dynamic size related APIs --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs that track and query dynamic size information.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicSize.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"

namespace clang {
namespace ento {

DefinedOrUnknownSVal getDynamicSize(ProgramStateRef State, const MemRegion *MR,
                                    SValBuilder &SVB) {
  return MR->getMemRegionManager().getStaticSize(MR, SVB);
}

DefinedOrUnknownSVal getDynamicElementCount(ProgramStateRef State,
                                            const MemRegion *MR,
                                            SValBuilder &SVB,
                                            QualType ElementTy) {
  MemRegionManager &MemMgr = MR->getMemRegionManager();
  ASTContext &Ctx = MemMgr.getContext();

  DefinedOrUnknownSVal Size = getDynamicSize(State, MR, SVB);
  SVal ElementSizeV = SVB.makeIntVal(
      Ctx.getTypeSizeInChars(ElementTy).getQuantity(), SVB.getArrayIndexType());

  SVal DivisionV =
      SVB.evalBinOp(State, BO_Div, Size, ElementSizeV, SVB.getArrayIndexType());

  return DivisionV.castAs<DefinedOrUnknownSVal>();
}

} // namespace ento
} // namespace clang
