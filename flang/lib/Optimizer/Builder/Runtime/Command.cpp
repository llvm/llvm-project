//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Command.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/command.h"

using namespace Fortran::runtime;

// Certain runtime intrinsics should only be run when select parameters of the
// intrisic are supplied. In certain cases one of these parameters may not be
// given, however the intrinsic needs to be run due to another required
// parameter being supplied. In this case the missing parameter is assigned to
// have an "absent" value. This typically happens in IntrinsicCall.cpp. For this
// reason the extra indirection with `isAbsent` is needed for testing whether a
// given parameter is actually present (so that parameters with "value" absent
// are not considered as present).
inline bool isAbsent(mlir::Value val) {
  return mlir::isa_and_nonnull<fir::AbsentOp>(val.getDefiningOp());
}

mlir::Value fir::runtime::genCommandArgumentCount(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  auto argumentCountFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentCount)>(loc, builder);
  return builder.create<fir::CallOp>(loc, argumentCountFunc).getResult(0);
}

mlir::Value fir::runtime::genGetCommand(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value command,
                                        mlir::Value length,
                                        mlir::Value errmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetCommand)>(loc, builder);
  mlir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(4));
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, command,
                                    length, errmsg, sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, runtimeFunc, args).getResult(0);
}

mlir::Value fir::runtime::genGetCommandArgument(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value number,
    mlir::Value value, mlir::Value length, mlir::Value errmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetCommandArgument)>(loc, builder);
  mlir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(5));
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, number, value,
                                    length, errmsg, sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, runtimeFunc, args).getResult(0);
}

mlir::Value fir::runtime::genGetEnvVariable(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value name, mlir::Value value,
                                            mlir::Value length,
                                            mlir::Value trimName,
                                            mlir::Value errmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetEnvVariable)>(loc, builder);
  mlir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(6));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, name, value, length, trimName, errmsg,
      sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, runtimeFunc, args).getResult(0);
}
