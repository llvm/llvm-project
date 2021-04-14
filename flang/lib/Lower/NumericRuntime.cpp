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

/// Helper function to recover the KIND from the REAL FIR type.
static int discoverRealKind(mlir::Type fltTy) {
  if (fltTy.isF16())
    return 2;
  if (fltTy.isBF16())
    return 3;
  if (fltTy.isF32())
    return 4;
  if (fltTy.isF64())
    return 8;
  if (fltTy.isF80())
    return 10;
  if (fltTy.isF128())
    return 16;
  llvm_unreachable("unexpected type");
}

/// Generate call to RRSpacing intrinsic runtime routine. 
mlir::Value
Fortran::lower::genRRSpacing(Fortran::lower::FirOpBuilder &builder, 
                           mlir::Location loc, mlir::Value x) { 
  mlir::FuncOp func;
  switch (discoverRealKind(x.getType())) {
  case 4:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing4)>(loc, builder);
    break;
  case 8:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing8)>(loc, builder);
    break;
#if LONG_DOUBLE == 80
  case 10:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing10)>(loc, builder);
    break;
#elif LONG_DOUBLE == 128
  case 16:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(RRSpacing16)>(loc, builder);
    break;
#endif
  default:
    llvm_unreachable("unsupported real kind");
  }

  auto fTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, fTy.getInput(0), x)
  };

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Spacing intrinsic runtime routine. 
mlir::Value
Fortran::lower::genSpacing(Fortran::lower::FirOpBuilder &builder, 
                           mlir::Location loc, mlir::Value x) { 
  mlir::FuncOp func;
  switch (discoverRealKind(x.getType())) {
  case 4:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing4)>(loc, builder);
    break;
  case 8:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing8)>(loc, builder);
    break;
#if LONG_DOUBLE == 80
  case 10:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing10)>(loc, builder);
    break;
#elif LONG_DOUBLE == 128
  case 16:
    func = Fortran::lower::getRuntimeFunc<mkRTKey(Spacing16)>(loc, builder);
    break;
#endif
  default:
    llvm_unreachable("unsupported real kind");
  }

  auto fTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, fTy.getInput(0), x)
  };

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
