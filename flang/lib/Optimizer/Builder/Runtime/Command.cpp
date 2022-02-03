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

void fir::runtime::genGetCommandArgument(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value number,
                                         mlir::Value value, mlir::Value length,
                                         mlir::Value status,
                                         mlir::Value errmsg) {
  auto argumentValueFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentValue)>(loc, builder);
  auto argumentLengthFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentLength)>(loc, builder);

  mlir::Value valueResult;
  // Run `ArgumentValue` intrinsic only if we have a "value" in either "VALUE",
  // "STATUS" or "ERRMSG" parameters.
  if (!isAbsent(value) || status || !isAbsent(errmsg)) {
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, argumentValueFunc.getType(), number, value, errmsg);
    valueResult =
        builder.create<fir::CallOp>(loc, argumentValueFunc, args).getResult(0);
  }

  // Only save result of `ArgumentValue` if "STATUS" parameter has been given
  if (status) {
    const mlir::Value statusLoaded = builder.create<fir::LoadOp>(loc, status);
    mlir::Value resultCast =
        builder.createConvert(loc, statusLoaded.getType(), valueResult);
    builder.create<fir::StoreOp>(loc, resultCast, status);
  }

  // Only run `ArgumentLength` intrinsic if "LENGTH" parameter provided
  if (length) {
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, argumentLengthFunc.getType(), number);
    mlir::Value result =
        builder.create<fir::CallOp>(loc, argumentLengthFunc, args).getResult(0);
    const mlir::Value valueLoaded = builder.create<fir::LoadOp>(loc, length);
    mlir::Value resultCast =
        builder.createConvert(loc, valueLoaded.getType(), result);
    builder.create<fir::StoreOp>(loc, resultCast, length);
  }
}

void fir::runtime::genGetEnvironmentVariable(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value name,
    mlir::Value value, mlir::Value length, mlir::Value status,
    mlir::Value trimName, mlir::Value errmsg) {
  auto valueFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(EnvVariableValue)>(loc, builder);
  auto lengthFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(EnvVariableLength)>(loc, builder);

  mlir::Value sourceFile;
  mlir::Value sourceLine;
  // We only need `sourceFile` and `sourceLine` variables when calling either
  // `EnvVariableValue` or `EnvVariableLength` below.
  if (!isAbsent(value) || status || !isAbsent(errmsg) || length) {
    sourceFile = fir::factory::locationToFilename(builder, loc);
    sourceLine = fir::factory::locationToLineNo(
        builder, loc, valueFunc.getType().getInput(5));
  }

  mlir::Value valueResult;
  // Run `EnvVariableValue` intrinsic only if we have a "value" in either
  // "VALUE", "STATUS" or "ERRMSG" parameters.
  if (!isAbsent(value) || status || !isAbsent(errmsg)) {
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, valueFunc.getType(), name, value, trimName, errmsg,
        sourceFile, sourceLine);
    valueResult =
        builder.create<fir::CallOp>(loc, valueFunc, args).getResult(0);
  }

  // Only save result of `EnvVariableValue` if "STATUS" parameter provided
  if (status) {
    const mlir::Value statusLoaded = builder.create<fir::LoadOp>(loc, status);
    mlir::Value resultCast =
        builder.createConvert(loc, statusLoaded.getType(), valueResult);
    builder.create<fir::StoreOp>(loc, resultCast, status);
  }

  // Only run `EnvVariableLength` intrinsic if "LENGTH" parameter provided
  if (length) {
    llvm::SmallVector<mlir::Value> args =
        fir::runtime::createArguments(builder, loc, lengthFunc.getType(), name,
                                      trimName, sourceFile, sourceLine);
    mlir::Value result =
        builder.create<fir::CallOp>(loc, lengthFunc, args).getResult(0);
    const mlir::Value lengthLoaded = builder.create<fir::LoadOp>(loc, length);
    mlir::Value resultCast =
        builder.createConvert(loc, lengthLoaded.getType(), result);
    builder.create<fir::StoreOp>(loc, resultCast, length);
  }
}
