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

// The real*10 and real*16 placeholders below are used to force the
// compilation of the real*10 and real*16 method names on systems that
// may not have them in their runtime library. This can occur in the
// case of cross compilation, for example.

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedRRSpacing10 {
  static constexpr const char *name = QuoteKey(RTNAME(RRSpacing10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of RRSpacing Intrinsic
struct ForcedRRSpacing16 {
  static constexpr const char *name = QuoteKey(RTNAME(RRSpacing16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Spacing Intrinsic
struct ForcedSpacing10 {
  static constexpr const char *name = QuoteKey(RTNAME(Spacing10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Spacing Intrinsic
struct ForcedSpacing16 {
  static constexpr const char *name = QuoteKey(RTNAME(Spacing16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

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
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedRRSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedRRSpacing16>(loc, builder);
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
  else if (fltTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedSpacing16>(loc, builder);
  else
    TODO(loc, "unsupported real kind in Spacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, funcTy.getInput(0), x)
  };

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
