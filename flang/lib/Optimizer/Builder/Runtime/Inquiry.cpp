//===-- Inquiry.h - generate inquiry runtime API calls ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/inquiry.h"

using namespace Fortran::runtime;

/// Generate call to `Lbound` runtime routine when the DIM argument is present.
mlir::Value fir::runtime::genLboundDim(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value array,
                                       mlir::Value dim) {
  mlir::FuncOp lboundFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(LboundDim)>(loc, builder);
  auto fTy = lboundFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, array, dim,
                                            sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, lboundFunc, args).getResult(0);
}

/// Generate call to `Ubound` runtime routine.  Calls to UBOUND with a DIM
/// argument get transformed into an expression equivalent to
/// SIZE() + LBOUND() - 1, so they don't have an intrinsic in the runtime.
void fir::runtime::genUbound(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value array,
                             mlir::Value kind) {
  mlir::FuncOp uboundFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Ubound)>(loc, builder);
  auto fTy = uboundFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox, array,
                                            kind, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, uboundFunc, args).getResult(0);
}

/// Generate call to `Size` runtime routine. This routine is a version when
/// the DIM argument is present.
mlir::Value fir::runtime::genSizeDim(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value array,
                                     mlir::Value dim) {
  mlir::FuncOp sizeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(SizeDim)>(loc, builder);
  auto fTy = sizeFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, array, dim,
                                            sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, sizeFunc, args).getResult(0);
}

/// Generate call to `Size` runtime routine. This routine is a version when
/// the DIM argument is absent.
mlir::Value fir::runtime::genSize(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value array) {
  mlir::FuncOp sizeFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Size)>(loc, builder);
  auto fTy = sizeFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, array,
                                            sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, sizeFunc, args).getResult(0);
}
