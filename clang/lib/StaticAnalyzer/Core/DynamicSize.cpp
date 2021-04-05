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

REGISTER_MAP_WITH_PROGRAMSTATE(DynamicSizeMap, const clang::ento::MemRegion *,
                               clang::ento::DefinedOrUnknownSVal)

namespace clang {
namespace ento {

DefinedOrUnknownSVal getDynamicSize(ProgramStateRef State, const MemRegion *MR,
                                    SValBuilder &SVB) {
  MR = MR->StripCasts();

  if (const DefinedOrUnknownSVal *Size = State->get<DynamicSizeMap>(MR))
    return *Size;

  return MR->getMemRegionManager().getStaticSize(MR, SVB);
}

DefinedOrUnknownSVal getElementSize(QualType Ty, SValBuilder &SVB) {
  return SVB.makeIntVal(SVB.getContext().getTypeSizeInChars(Ty).getQuantity(),
                        SVB.getArrayIndexType());
}

DefinedOrUnknownSVal getDynamicElementCount(ProgramStateRef State,
                                            const MemRegion *MR,
                                            SValBuilder &SVB,
                                            QualType ElementTy) {
  MR = MR->StripCasts();

  DefinedOrUnknownSVal Size = getDynamicSize(State, MR, SVB);
  SVal ElementSize = getElementSize(ElementTy, SVB);

  SVal ElementCount =
      SVB.evalBinOp(State, BO_Div, Size, ElementSize, SVB.getArrayIndexType());

  return ElementCount.castAs<DefinedOrUnknownSVal>();
}

SVal getDynamicSizeWithOffset(ProgramStateRef State, SVal BufV) {
  SValBuilder &SvalBuilder = State->getStateManager().getSValBuilder();
  const MemRegion *MRegion = BufV.getAsRegion();
  if (!MRegion)
    return UnknownVal();
  RegionOffset Offset = MRegion->getAsOffset();
  if (Offset.hasSymbolicOffset())
    return UnknownVal();
  const MemRegion *BaseRegion = MRegion->getBaseRegion();
  if (!BaseRegion)
    return UnknownVal();

  NonLoc OffsetInBytes = SvalBuilder.makeArrayIndex(
      Offset.getOffset() /
      MRegion->getMemRegionManager().getContext().getCharWidth());
  DefinedOrUnknownSVal ExtentInBytes =
      getDynamicSize(State, BaseRegion, SvalBuilder);

  return SvalBuilder.evalBinOp(State, BinaryOperator::Opcode::BO_Sub,
                               ExtentInBytes, OffsetInBytes,
                               SvalBuilder.getArrayIndexType());
}

ProgramStateRef setDynamicSize(ProgramStateRef State, const MemRegion *MR,
                               DefinedOrUnknownSVal Size, SValBuilder &SVB) {
  MR = MR->StripCasts();

  if (Size.isUnknown())
    return State;

  return State->set<DynamicSizeMap>(MR->StripCasts(), Size);
}

} // namespace ento
} // namespace clang
