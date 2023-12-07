//===-- Transformational.cpp ------------------------------------*- C++ -*-===//
// Generate transformational intrinsic runtime API calls.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Runtime/matmul-transpose.h"
#include "flang/Runtime/matmul.h"
#include "flang/Runtime/transformational.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;

/// Placeholder for real*10 version of BesselJn intrinsic.
struct ForcedBesselJn_10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselJn_10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(
          ctx, {boxTy, intTy, intTy, ty, ty, ty, strTy, intTy}, {noneTy});
    };
  }
};

/// Placeholder for real*16 version of BesselJn intrinsic.
struct ForcedBesselJn_16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselJn_16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(
          ctx, {boxTy, intTy, intTy, ty, ty, ty, strTy, intTy}, {noneTy});
    };
  }
};

/// Placeholder for real*10 version of BesselJn intrinsic when `x == 0.0`.
struct ForcedBesselJnX0_10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselJnX0_10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(ctx, {boxTy, intTy, intTy, strTy, intTy},
                                     {noneTy});
    };
  }
};

/// Placeholder for real*16 version of BesselJn intrinsic when `x == 0.0`.
struct ForcedBesselJnX0_16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselJnX0_16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(ctx, {boxTy, intTy, intTy, strTy, intTy},
                                     {noneTy});
    };
  }
};

/// Placeholder for real*10 version of BesselYn intrinsic.
struct ForcedBesselYn_10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselYn_10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(
          ctx, {boxTy, intTy, intTy, ty, ty, ty, strTy, intTy}, {noneTy});
    };
  }
};

/// Placeholder for real*16 version of BesselYn intrinsic.
struct ForcedBesselYn_16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselYn_16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(
          ctx, {boxTy, intTy, intTy, ty, ty, ty, strTy, intTy}, {noneTy});
    };
  }
};

/// Placeholder for real*10 version of BesselYn intrinsic when `x == 0.0`.
struct ForcedBesselYnX0_10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselYnX0_10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(ctx, {boxTy, intTy, intTy, strTy, intTy},
                                     {noneTy});
    };
  }
};

/// Placeholder for real*16 version of BesselYn intrinsic when `x == 0.0`.
struct ForcedBesselYnX0_16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(BesselYnX0_16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto boxTy =
          fir::runtime::getModel<Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 32);
      auto noneTy = mlir::NoneType::get(ctx);
      return mlir::FunctionType::get(ctx, {boxTy, intTy, intTy, strTy, intTy},
                                     {noneTy});
    };
  }
};

/// Generate call to `BesselJn` intrinsic.
void fir::runtime::genBesselJn(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value resultBox, mlir::Value n1,
                               mlir::Value n2, mlir::Value x, mlir::Value bn2,
                               mlir::Value bn2_1) {
  mlir::func::FuncOp func;
  auto xTy = x.getType();

  if (xTy.isF16() || xTy.isBF16())
    TODO(loc, "half-precision BESSEL_JN");
  else if (xTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselJn_4)>(loc, builder);
  else if (xTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselJn_8)>(loc, builder);
  else if (xTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedBesselJn_10>(loc, builder);
  else if (xTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedBesselJn_16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in BESSEL_JN");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(7));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, n1, n2, x,
                                    bn2, bn2_1, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `BesselJn` intrinsic. This is used when `x == 0.0`.
void fir::runtime::genBesselJnX0(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Type xTy, mlir::Value resultBox,
                                 mlir::Value n1, mlir::Value n2) {
  mlir::func::FuncOp func;

  if (xTy.isF16() || xTy.isBF16())
    TODO(loc, "half-precision BESSEL_JN");
  else if (xTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselJnX0_4)>(loc, builder);
  else if (xTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselJnX0_8)>(loc, builder);
  else if (xTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedBesselJnX0_10>(loc, builder);
  else if (xTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedBesselJnX0_16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in BESSEL_JN");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox, n1,
                                            n2, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `BesselYn` intrinsic.
void fir::runtime::genBesselYn(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value resultBox, mlir::Value n1,
                               mlir::Value n2, mlir::Value x, mlir::Value bn1,
                               mlir::Value bn1_1) {
  mlir::func::FuncOp func;
  auto xTy = x.getType();

  if (xTy.isF16() || xTy.isBF16())
    TODO(loc, "half-precision BESSEL_YN");
  else if (xTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselYn_4)>(loc, builder);
  else if (xTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselYn_8)>(loc, builder);
  else if (xTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedBesselYn_10>(loc, builder);
  else if (xTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedBesselYn_16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in BESSEL_YN");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(7));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, n1, n2, x,
                                    bn1, bn1_1, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `BesselYn` intrinsic. This is used when `x == 0.0`.
void fir::runtime::genBesselYnX0(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Type xTy, mlir::Value resultBox,
                                 mlir::Value n1, mlir::Value n2) {
  mlir::func::FuncOp func;

  if (xTy.isF16() || xTy.isBF16())
    TODO(loc, "half-precision BESSEL_YN");
  else if (xTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselYnX0_4)>(loc, builder);
  else if (xTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(BesselYnX0_8)>(loc, builder);
  else if (xTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedBesselYnX0_10>(loc, builder);
  else if (xTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedBesselYnX0_16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in BESSEL_YN");

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox, n1,
                                            n2, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Cshift intrinsic
void fir::runtime::genCshift(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value shiftBox, mlir::Value dimBox) {
  auto cshiftFunc = fir::runtime::getRuntimeFunc<mkRTKey(Cshift)>(loc, builder);
  auto fTy = cshiftFunc.getFunctionType();
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
  auto fTy = cshiftFunc.getFunctionType();

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
  auto fTy = eoshiftFunc.getFunctionType();
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
  auto fTy = eoshiftFunc.getFunctionType();

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
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, matrixABox,
                                    matrixBBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to MatmulTranspose intrinsic runtime routine.
void fir::runtime::genMatmulTranspose(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value resultBox,
                                      mlir::Value matrixABox,
                                      mlir::Value matrixBBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(MatmulTranspose)>(loc, builder);
  auto fTy = func.getFunctionType();
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
  auto fTy = packFunc.getFunctionType();
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
  auto fTy = func.getFunctionType();
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
  auto fTy = func.getFunctionType();
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
  auto fTy = func.getFunctionType();
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
  auto fTy = unpackFunc.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, vectorBox,
                                    maskBox, fieldBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, unpackFunc, args);
}
