//===-- Stop.h - generate stop runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/stop.h"

using namespace Fortran::runtime;

void fir::runtime::genExit(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value status) {
  auto exitFunc = fir::runtime::getRuntimeFunc<mkRTKey(Exit)>(loc, builder);
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, exitFunc.getType(), status);
  builder.create<fir::CallOp>(loc, exitFunc, args);
}
