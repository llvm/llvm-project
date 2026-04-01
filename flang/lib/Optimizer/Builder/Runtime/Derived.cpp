//===-- Derived.cpp -- derived type runtime API ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/derived-api.h"
#include "flang/Runtime/pointer.h"

using namespace Fortran::runtime;

void fir::runtime::genDerivedTypeInitialize(fir::FirOpBuilder &builder,
                                            aiir::Location loc,
                                            aiir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Initialize)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, box, sourceFile,
                                            sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDerivedTypeInitializeClone(fir::FirOpBuilder &builder,
                                                 aiir::Location loc,
                                                 aiir::Value newBox,
                                                 aiir::Value box) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(InitializeClone)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, newBox, box,
                                            sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDerivedTypeDestroy(fir::FirOpBuilder &builder,
                                         aiir::Location loc, aiir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Destroy)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDerivedTypeFinalize(fir::FirOpBuilder &builder,
                                          aiir::Location loc, aiir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Finalize)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, box, sourceFile,
                                            sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genDerivedTypeDestroyWithoutFinalization(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(DestroyWithoutFinalization)>(
      loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genNullifyDerivedType(fir::FirOpBuilder &builder,
                                         aiir::Location loc, aiir::Value box,
                                         fir::RecordType derivedType,
                                         unsigned rank) {
  aiir::Value typeDesc =
      fir::TypeDescOp::create(builder, loc, aiir::TypeAttr::get(derivedType));
  aiir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerNullifyDerived)>(loc,
                                                                   builder);
  llvm::ArrayRef<aiir::Type> inputTypes = callee.getFunctionType().getInputs();
  llvm::SmallVector<aiir::Value> args;
  args.push_back(builder.createConvert(loc, inputTypes[0], box));
  args.push_back(builder.createConvert(loc, inputTypes[1], typeDesc));
  aiir::Value rankCst = builder.createIntegerConstant(loc, inputTypes[2], rank);
  aiir::Value c0 = builder.createIntegerConstant(loc, inputTypes[3], 0);
  args.push_back(rankCst);
  args.push_back(c0);
  fir::CallOp::create(builder, loc, callee, args);
}

aiir::Value fir::runtime::genSameTypeAs(fir::FirOpBuilder &builder,
                                        aiir::Location loc, aiir::Value a,
                                        aiir::Value b) {
  aiir::func::FuncOp sameTypeAsFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(SameTypeAs)>(loc, builder);
  auto fTy = sameTypeAsFunc.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, a, b);
  return fir::CallOp::create(builder, loc, sameTypeAsFunc, args).getResult(0);
}

aiir::Value fir::runtime::genExtendsTypeOf(fir::FirOpBuilder &builder,
                                           aiir::Location loc, aiir::Value a,
                                           aiir::Value mold) {
  aiir::func::FuncOp extendsTypeOfFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ExtendsTypeOf)>(loc, builder);
  auto fTy = extendsTypeOfFunc.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, a, mold);
  return fir::CallOp::create(builder, loc, extendsTypeOfFunc, args)
      .getResult(0);
}
