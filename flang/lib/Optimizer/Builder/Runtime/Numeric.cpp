//===-- Numeric.cpp -- runtime API for numeric intrinsics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Numeric.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/numeric.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

// The real*10 and real*16 placeholders below are used to force the
// compilation of the real*10 and real*16 method names on systems that
// may not have them in their runtime library. This can occur in the
// case of cross compilation, for example.

/// Placeholder for real*10 version of Exponent Intrinsic
struct ForcedExponent10_4 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent10_4));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 32);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

struct ForcedExponent10_8 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent10_8));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

/// Placeholder for real*16 version of Exponent Intrinsic
struct ForcedExponent16_4 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent16_4));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 32);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

struct ForcedExponent16_8 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Exponent16_8));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, fltTy, intTy);
    };
  }
};

/// Placeholder for real*10 version of Fraction Intrinsic
struct ForcedFraction10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Fraction10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Fraction Intrinsic
struct ForcedFraction16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Fraction16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Nearest Intrinsic
struct ForcedNearest10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Nearest10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto boolTy = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(ctx, {fltTy, boolTy}, {fltTy});
    };
  }
};

/// Placeholder for real*16 version of Nearest Intrinsic
struct ForcedNearest16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Nearest16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto boolTy = mlir::IntegerType::get(ctx, 1);
      return mlir::FunctionType::get(ctx, {fltTy, boolTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedRRSpacing10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(RRSpacing10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of RRSpacing Intrinsic
struct ForcedRRSpacing16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(RRSpacing16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Scale Intrinsic
struct ForcedScale10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Scale10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*16 version of Scale Intrinsic
struct ForcedScale16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Scale16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedSetExponent10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SetExponent10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of RRSpacing Intrinsic
struct ForcedSetExponent16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SetExponent16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto intTy = mlir::IntegerType::get(ctx, 64);
      return mlir::FunctionType::get(ctx, {fltTy, intTy}, {fltTy});
    };
  }
};

/// Placeholder for real*10 version of Spacing Intrinsic
struct ForcedSpacing10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Spacing10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Spacing Intrinsic
struct ForcedSpacing16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(Spacing16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Generate call to Exponent instrinsic runtime routine.
mlir::Value fir::runtime::genExponent(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type resultType,
                                      mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32()) {
    if (resultType.isInteger(32))
      func = fir::runtime::getRuntimeFunc<mkRTKey(Exponent4_4)>(loc, builder);
    else if (resultType.isInteger(64))
      func = fir::runtime::getRuntimeFunc<mkRTKey(Exponent4_8)>(loc, builder);
  } else if (fltTy.isF64()) {
    if (resultType.isInteger(32))
      func = fir::runtime::getRuntimeFunc<mkRTKey(Exponent8_4)>(loc, builder);
    else if (resultType.isInteger(64))
      func = fir::runtime::getRuntimeFunc<mkRTKey(Exponent8_8)>(loc, builder);
  } else if (fltTy.isF80()) {
    if (resultType.isInteger(32))
      func = fir::runtime::getRuntimeFunc<ForcedExponent10_4>(loc, builder);
    else if (resultType.isInteger(64))
      func = fir::runtime::getRuntimeFunc<ForcedExponent10_8>(loc, builder);
  } else if (fltTy.isF128()) {
    if (resultType.isInteger(32))
      func = fir::runtime::getRuntimeFunc<ForcedExponent16_4>(loc, builder);
    else if (resultType.isInteger(64))
      func = fir::runtime::getRuntimeFunc<ForcedExponent16_8>(loc, builder);
  } else
    fir::emitFatalError(loc, "unsupported real kind in Exponent lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Fraction instrinsic runtime routine.
mlir::Value fir::runtime::genFraction(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Fraction4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Fraction8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedFraction10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedFraction16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported real kind in Fraction lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Nearest intrinsic runtime routine.
mlir::Value fir::runtime::genNearest(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value x,
                                     mlir::Value s) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Nearest4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Nearest8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedNearest10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedNearest16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported REAL kind in Nearest lowering");

  auto funcTy = func.getType();

  mlir::Type sTy = s.getType();
  mlir::Value zero = builder.createRealZeroConstant(loc, sTy);
  auto cmp = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OGT, s, zero);

  mlir::Type boolTy = mlir::IntegerType::get(builder.getContext(), 1);
  mlir::Value False = builder.createIntegerConstant(loc, boolTy, 0);
  mlir::Value True = builder.createIntegerConstant(loc, boolTy, 1);

  mlir::Value positive =
      builder.create<mlir::arith::SelectOp>(loc, cmp, True, False);
  auto args = fir::runtime::createArguments(builder, loc, funcTy, x, positive);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to RRSpacing intrinsic runtime routine.
mlir::Value fir::runtime::genRRSpacing(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(RRSpacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(RRSpacing8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedRRSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedRRSpacing16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported real kind in RRSpacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Scale intrinsic runtime routine.
mlir::Value fir::runtime::genScale(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value x,
                                   mlir::Value i) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Scale4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Scale8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedScale10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedScale16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported REAL kind in Scale lowering");

  auto funcTy = func.getType();
  auto args = fir::runtime::createArguments(builder, loc, funcTy, x, i);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Set_exponent instrinsic runtime routine.
mlir::Value fir::runtime::genSetExponent(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value x,
                                         mlir::Value i) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(SetExponent4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(SetExponent8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedSetExponent10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedSetExponent16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported real kind in Fraction lowering");

  auto funcTy = func.getType();
  auto args = fir::runtime::createArguments(builder, loc, funcTy, x, i);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Spacing intrinsic runtime routine.
mlir::Value fir::runtime::genSpacing(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value x) {
  mlir::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Spacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Spacing8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedSpacing16>(loc, builder);
  else
    fir::emitFatalError(loc, "unsupported real kind in Spacing lowering");

  auto funcTy = func.getType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
