//===-- Allocatable.cpp -- generate allocatable runtime API calls----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/allocatable.h"

using namespace Fortran::runtime;

aiir::Value fir::runtime::genMoveAlloc(fir::FirOpBuilder &builder,
                                       aiir::Location loc, aiir::Value to,
                                       aiir::Value from, aiir::Value hasStat,
                                       aiir::Value errMsg) {
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(MoveAlloc)>(loc, builder)};
  aiir::FunctionType fTy{func.getFunctionType()};
  aiir::Value sourceFile{fir::factory::locationToFilename(builder, loc)};
  aiir::Value sourceLine{
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(6))};
  aiir::Value declaredTypeDesc;
  if (fir::isPolymorphicType(from.getType()) &&
      !fir::isUnlimitedPolymorphicType(from.getType())) {
    fir::ClassType clTy =
        aiir::dyn_cast<fir::ClassType>(fir::dyn_cast_ptrEleTy(from.getType()));
    aiir::Type derivedType = fir::unwrapInnerType(clTy.getEleTy());
    declaredTypeDesc =
        fir::TypeDescOp::create(builder, loc, aiir::TypeAttr::get(derivedType));
  } else {
    declaredTypeDesc = builder.createNullConstant(loc);
  }
  llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, to, from, declaredTypeDesc, hasStat, errMsg,
      sourceFile, sourceLine)};

  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genAllocatableApplyMold(fir::FirOpBuilder &builder,
                                           aiir::Location loc, aiir::Value desc,
                                           aiir::Value mold, int rank) {
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(AllocatableApplyMold)>(loc,
                                                                  builder)};
  aiir::FunctionType fTy = func.getFunctionType();
  aiir::Value rankVal =
      builder.createIntegerConstant(loc, fTy.getInput(2), rank);
  llvm::SmallVector<aiir::Value> args{
      fir::runtime::createArguments(builder, loc, fTy, desc, mold, rankVal)};
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genAllocatableSetBounds(fir::FirOpBuilder &builder,
                                           aiir::Location loc, aiir::Value desc,
                                           aiir::Value dimIndex,
                                           aiir::Value lowerBound,
                                           aiir::Value upperBound) {
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(AllocatableSetBounds)>(loc,
                                                                  builder)};
  aiir::FunctionType fTy{func.getFunctionType()};
  llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, desc, dimIndex, lowerBound, upperBound)};
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genAllocatableAllocate(fir::FirOpBuilder &builder,
                                          aiir::Location loc, aiir::Value desc,
                                          aiir::Value hasStat,
                                          aiir::Value errMsg) {
  aiir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(AllocatableAllocate)>(loc, builder)};
  aiir::FunctionType fTy{func.getFunctionType()};
  aiir::Value asyncObject = builder.createNullConstant(loc);
  aiir::Value sourceFile{fir::factory::locationToFilename(builder, loc)};
  aiir::Value sourceLine{
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5))};
  if (!hasStat)
    hasStat = builder.createBool(loc, false);
  if (!errMsg) {
    aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    errMsg = fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  }
  aiir::Value deviceInit = builder.createBool(loc, false);
  llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, desc, asyncObject, hasStat, errMsg, sourceFile,
      sourceLine, deviceInit)};
  fir::CallOp::create(builder, loc, func, args);
}
