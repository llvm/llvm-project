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
#include "flang/Runtime/extensions.h"

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

aiir::Value fir::runtime::genCommandArgumentCount(fir::FirOpBuilder &builder,
                                                  aiir::Location loc) {
  auto argumentCountFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentCount)>(loc, builder);
  return fir::CallOp::create(builder, loc, argumentCountFunc).getResult(0);
}

aiir::Value fir::runtime::genGetCommand(fir::FirOpBuilder &builder,
                                        aiir::Location loc, aiir::Value command,
                                        aiir::Value length,
                                        aiir::Value errmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetCommand)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(4));
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, command,
                                    length, errmsg, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

aiir::Value fir::runtime::genGetPID(fir::FirOpBuilder &builder,
                                    aiir::Location loc) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetPID)>(loc, builder);

  return fir::CallOp::create(builder, loc, runtimeFunc).getResult(0);
}

aiir::Value fir::runtime::genGetCommandArgument(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value number,
    aiir::Value value, aiir::Value length, aiir::Value errmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetCommandArgument)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(5));
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, number, value,
                                    length, errmsg, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

aiir::Value fir::runtime::genGetEnvVariable(fir::FirOpBuilder &builder,
                                            aiir::Location loc,
                                            aiir::Value name, aiir::Value value,
                                            aiir::Value length,
                                            aiir::Value trimName,
                                            aiir::Value errmsg) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(GetEnvVariable)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(6));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, name, value, length, trimName, errmsg,
      sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, runtimeFunc, args).getResult(0);
}

aiir::Value fir::runtime::genGetCwd(fir::FirOpBuilder &builder,
                                    aiir::Location loc, aiir::Value cwd) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(GetCwd)>(loc, builder);
  auto runtimeFuncTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(2));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, cwd, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

aiir::Value fir::runtime::genHostnm(fir::FirOpBuilder &builder,
                                    aiir::Location loc, aiir::Value res) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(Hostnm)>(loc, builder);
  auto runtimeFuncTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(2));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, res, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genPerror(fir::FirOpBuilder &builder, aiir::Location loc,
                             aiir::Value string) {
  auto runtimeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Perror)>(loc, builder);
  aiir::FunctionType runtimeFuncTy = runtimeFunc.getFunctionType();
  llvm::SmallVector<aiir::Value> args =
      fir::runtime::createArguments(builder, loc, runtimeFuncTy, string);
  fir::CallOp::create(builder, loc, runtimeFunc, args);
}

aiir::Value fir::runtime::genPutEnv(fir::FirOpBuilder &builder,
                                    aiir::Location loc, aiir::Value str,
                                    aiir::Value strLength) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PutEnv)>(loc, builder);
  auto runtimeFuncTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(1));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, str, strLength, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

aiir::Value fir::runtime::genUnlink(fir::FirOpBuilder &builder,
                                    aiir::Location loc, aiir::Value path,
                                    aiir::Value pathLength) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(Unlink)>(loc, builder);
  auto runtimeFuncTy = func.getFunctionType();
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, runtimeFuncTy.getInput(1));
  llvm::SmallVector<aiir::Value> args = fir::runtime::createArguments(
      builder, loc, runtimeFuncTy, path, pathLength, sourceFile, sourceLine);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}
