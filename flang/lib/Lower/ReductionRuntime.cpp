//===-- ReductionRuntime.cpp -- runtime for reduction intrinsics -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ReductionRuntime.h"
#include "../../runtime/reduction.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Generate calls to reduction intrinsics such as All and Any.
/// These are the descriptor based implementations that take two 
/// arguments (mask, dim).
template <typename FN>
static void
genRed2Args(FN func, Fortran::lower::FirOpBuilder &builder,
            mlir::Location loc, mlir::Value resultBox,
            mlir::Value maskBox, mlir::Value dim) {
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
       Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(4));

  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, fTy.getInput(0), resultBox),
    builder.createConvert(loc, fTy.getInput(1), maskBox),
    builder.createConvert(loc, fTy.getInput(2), dim),
    builder.createConvert(loc, fTy.getInput(3), sourceFile),
    builder.createConvert(loc, fTy.getInput(4), sourceLine) 
  };
   
  builder.create<fir::CallOp>(loc, func, args);

}
/// Generate call to all runtime routine.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void
Fortran::lower::genAllDescriptor(Fortran::lower::FirOpBuilder &builder, 
                                 mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value maskBox,
                                 mlir::Value dim) {
  auto allFunc = Fortran::lower::getRuntimeFunc<mkRTKey(AllDim)>(loc, builder);
  genRed2Args(allFunc, builder, loc, resultBox, maskBox, dim);
}


/// Generate call to any runtime routine.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void
Fortran::lower::genAnyDescriptor(Fortran::lower::FirOpBuilder &builder, 
                                 mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value maskBox,
                                 mlir::Value dim) {
  auto allFunc = Fortran::lower::getRuntimeFunc<mkRTKey(AnyDim)>(loc, builder);
  genRed2Args(allFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to All intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value
Fortran::lower::genAll(Fortran::lower::FirOpBuilder &builder, 
                       mlir::Location loc, mlir::Value maskBox, 
                       mlir::Value dim) {
  auto allFunc = Fortran::lower::getRuntimeFunc<mkRTKey(All)>(loc, builder);
  auto fTy = allFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
       Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(2));
     
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, fTy.getInput(0), maskBox),
    builder.createConvert(loc, fTy.getInput(1), sourceFile),
    builder.createConvert(loc, fTy.getInput(2), sourceLine), 
    builder.createConvert(loc, fTy.getInput(3), dim)
  };
   
  return builder.create<fir::CallOp>(loc, allFunc, args).getResult(0);
}

/// Generate call to Any intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value
Fortran::lower::genAny(Fortran::lower::FirOpBuilder &builder, 
                       mlir::Location loc, mlir::Value maskBox, 
                       mlir::Value dim) {
  auto anyFunc = Fortran::lower::getRuntimeFunc<mkRTKey(Any)>(loc, builder);
  auto fTy = anyFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
       Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(2));
     
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, fTy.getInput(0), maskBox),
    builder.createConvert(loc, fTy.getInput(1), sourceFile),
    builder.createConvert(loc, fTy.getInput(2), sourceLine), 
    builder.createConvert(loc, fTy.getInput(3), dim)
  };
   
  return builder.create<fir::CallOp>(loc, anyFunc, args).getResult(0);
}
