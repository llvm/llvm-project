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

mlir::Value fir::runtime::genMoveAlloc(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value to,
                                       mlir::Value from, mlir::Value hasStat,
                                       mlir::Value errMsg) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(MoveAlloc)>(loc, builder)};
  mlir::FunctionType fTy{func.getFunctionType()};
  mlir::Value sourceFile{fir::factory::locationToFilename(builder, loc)};
  mlir::Value sourceLine{
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(6))};
  mlir::Value declaredTypeDesc;
  if (fir::isPolymorphicType(from.getType()) &&
      !fir::isUnlimitedPolymorphicType(from.getType())) {
    fir::ClassType clTy =
        fir::dyn_cast_ptrEleTy(from.getType()).dyn_cast<fir::ClassType>();
    mlir::Type derivedType = fir::unwrapInnerType(clTy.getEleTy());
    declaredTypeDesc =
        builder.create<fir::TypeDescOp>(loc, mlir::TypeAttr::get(derivedType));
  } else {
    declaredTypeDesc = builder.createNullConstant(loc);
  }
  llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, to, from, declaredTypeDesc, hasStat, errMsg,
      sourceFile, sourceLine)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

void fir::runtime::genAllocatableApplyMold(fir::FirOpBuilder &builder,
                                           mlir::Location loc, mlir::Value desc,
                                           mlir::Value mold, int rank) {
  mlir::func::FuncOp func{
      fir::runtime::getRuntimeFunc<mkRTKey(AllocatableApplyMold)>(loc,
                                                                  builder)};
  mlir::FunctionType fTy = func.getFunctionType();
  mlir::Value rankVal =
      builder.createIntegerConstant(loc, fTy.getInput(2), rank);
  llvm::SmallVector<mlir::Value> args{
      fir::runtime::createArguments(builder, loc, fTy, desc, mold, rankVal)};
  builder.create<fir::CallOp>(loc, func, args);
}
