//===-- NumericRuntime.cpp -- runtime for numeric intrinsics -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../runtime/numeric.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "flang/Lower/NumericRuntime.h"

using namespace Fortran::runtime;

/// Generate call to RRSpacing intrinsic runtime routine. 
mlir::Value
Fortran::lower::genRRSpacing(Fortran::lower::FirOpBuilder &builder, 
                           mlir::Location loc, mlir::Value x) { 
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing8)>(loc, builder);
#if LONG_DOUBLE == 80
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing10)>(loc, builder);
#elif LONG_DOUBLE == 128
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing16)>(loc, builder);
#endif
  else
    TODO(loc, "unsupported real kind in RRSpacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, funcTy.getInput(0), x)
  };

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Spacing intrinsic runtime routine. 
mlir::Value
Fortran::lower::genSpacing(Fortran::lower::FirOpBuilder &builder, 
                           mlir::Location loc, mlir::Value x) { 
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing8)>(loc, builder);
#if LONG_DOUBLE == 80
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing10)>(loc, builder);
#elif LONG_DOUBLE == 128
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing16)>(loc, builder);
#endif
  else
    TODO(loc, "unsupported real kind in Spacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, funcTy.getInput(0), x)
  };

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
