//===-- Execute.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Execute.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/execute.h"

using namespace Fortran::runtime;

// Certain runtime intrinsics should only be run when select parameters of the
// intrisic are supplied. In certain cases one of these parameters may not be
// given, however the intrinsic needs to be run due to another required
// parameter being supplied. In this case the missing parameter is assigned to
// have an "absent" value. This typically happens in IntrinsicCall.cpp. For this
// reason the extra indirection with `isAbsent` is needed for testing whether a
// given parameter is actually present (so that parameters with "value" absent
// are not considered as present).
inline bool isAbsent(aiir::Value val) {
  return aiir::isa_and_nonnull<fir::AbsentOp>(val.getDefiningOp());
}

void fir::runtime::genExecuteCommandLine(fir::FirOpBuilder &builder,
                                         aiir::Location loc,
                                         aiir::Value command, aiir::Value wait,
                                         aiir::Value exitstat,
                                         aiir::Value cmdstat,
                                         aiir::Value cmdmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ExecuteCommandLine)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(6));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, command, wait, exitstat, cmdstat, cmdmsg,
      sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, runtimeFunc, args);
}
