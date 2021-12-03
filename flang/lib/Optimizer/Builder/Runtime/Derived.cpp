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
#include "flang/Runtime/derived-api.h"

using namespace Fortran::runtime;

void fir::runtime::genDerivedTypeInitialize(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Initialize)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, box, sourceFile,
                                            sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::genDerivedTypeDestroy(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Destroy)>(loc, builder);
  auto fTy = func.getType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  builder.create<fir::CallOp>(loc, func, args);
}
