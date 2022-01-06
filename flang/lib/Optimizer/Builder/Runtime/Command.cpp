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

  auto isPresent = [&](mlir::Value val) -> bool {
    return !mlir::isa_and_nonnull<fir::AbsentOp>(val.getDefiningOp());
  };

  mlir::Value valueResult;
  // Run `ArgumentValue` intrisc only if we have either "value", "status" or
  // "errmsg" `ArgumentValue` "requires" existing values for its arguments
  // "value" and "errmsg". So in the case they aren't given, but the user has
  // requested "status", we have to assign "absent" values to them before
  // calling `ArgumentValue`. This happens in IntrinsicCall.cpp. For this reason
  // we need extra indirection with `isPresent` for testing whether "value" or
  // "errmsg" is present.
  if (isPresent(value) || status || isPresent(errmsg)) {
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, argumentValueFunc.getType(), number, value, errmsg);
    valueResult =
        builder.create<fir::CallOp>(loc, argumentValueFunc, args).getResult(0);
  }

  // Only save result of ArgumentValue if "status" parameter has been given
  if (status) {
    const mlir::Value statusLoaded = builder.create<fir::LoadOp>(loc, status);
    mlir::Value resultCast =
        builder.createConvert(loc, statusLoaded.getType(), valueResult);
    builder.create<fir::StoreOp>(loc, resultCast, status);
  }

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
