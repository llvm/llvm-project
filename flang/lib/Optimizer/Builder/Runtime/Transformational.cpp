//===-- Transformational.cpp ------------------------------------*- C++ -*-===//
// Generate transformational intrinsic runtime API calls.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/matmul.h"
#include "flang/Runtime/transformational.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;

/// Generate call to Cshift intrinsic
void fir::runtime::genCshift(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value shiftBox, mlir::Value dimBox) {
  auto cshiftFunc = fir::runtime::getRuntimeFunc<mkRTKey(Cshift)>(loc, builder);
  auto fTy = cshiftFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                    shiftBox, dimBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, cshiftFunc, args);
}

/// Generate call to the vector version of the Cshift intrinsic
void fir::runtime::genCshiftVector(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value resultBox,
                                   mlir::Value arrayBox, mlir::Value shiftBox) {
  auto cshiftFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(CshiftVector)>(loc, builder);
  auto fTy = cshiftFunc.getType();

  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, shiftBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, cshiftFunc, args);
}

/// Generate call to Eoshift intrinsic
void fir::runtime::genEoshift(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value arrayBox,
                              mlir::Value shiftBox, mlir::Value boundBox,
                              mlir::Value dimBox) {
  auto eoshiftFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Eoshift)>(loc, builder);
  auto fTy = eoshiftFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(6));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            arrayBox, shiftBox, boundBox,
                                            dimBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, eoshiftFunc, args);
}

/// Generate call to the vector version of the Eoshift intrinsic
void fir::runtime::genEoshiftVector(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value resultBox,
                                    mlir::Value arrayBox, mlir::Value shiftBox,
                                    mlir::Value boundBox) {
  auto eoshiftFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(EoshiftVector)>(loc, builder);
  auto fTy = eoshiftFunc.getType();

  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));

  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                    shiftBox, boundBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, eoshiftFunc, args);
}

/// Generate call to Matmul intrinsic runtime routine.
void fir::runtime::genMatmul(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value matrixABox,
                             mlir::Value matrixBBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Matmul)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, matrixABox,
                                    matrixBBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Pack intrinsic runtime routine.
void fir::runtime::genPack(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value resultBox, mlir::Value arrayBox,
                           mlir::Value maskBox, mlir::Value vectorBox) {
  auto packFunc = fir::runtime::getRuntimeFunc<mkRTKey(Pack)>(loc, builder);
  auto fTy = packFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                    maskBox, vectorBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, packFunc, args);
}

/// Generate call to Reshape intrinsic runtime routine.
void fir::runtime::genReshape(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value sourceBox,
                              mlir::Value shapeBox, mlir::Value padBox,
                              mlir::Value orderBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Reshape)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(6));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            sourceBox, shapeBox, padBox,
                                            orderBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Spread intrinsic runtime routine.
void fir::runtime::genSpread(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value sourceBox,
                             mlir::Value dim, mlir::Value ncopies) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Spread)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, sourceBox,
                                    dim, ncopies, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Transpose intrinsic runtime routine.
void fir::runtime::genTranspose(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value sourceBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Transpose)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            sourceBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Unpack intrinsic runtime routine.
void fir::runtime::genUnpack(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value vectorBox,
                             mlir::Value maskBox, mlir::Value fieldBox) {
  auto unpackFunc = fir::runtime::getRuntimeFunc<mkRTKey(Unpack)>(loc, builder);
  auto fTy = unpackFunc.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, vectorBox,
                                    maskBox, fieldBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, unpackFunc, args);
}
