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
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Runtime/numeric.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;

// The real*10 and real*16 placeholders below are used to force the
// compilation of the real*10 and real*16 method names on systems that
// may not have them in their runtime library. This can occur in the
// case of cross compilation, for example.

/// Placeholder for real*10 version of ErfcScaled Intrinsic
struct ForcedErfcScaled10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ErfcScaled10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of ErfcScaled Intrinsic
struct ForcedErfcScaled16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ErfcScaled16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

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

/// Placeholder for real*10 version of Mod Intrinsic
struct ForcedMod10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ModReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {fltTy, fltTy, strTy, intTy},
                                     {fltTy});
    };
  }
};

/// Placeholder for real*16 version of Mod Intrinsic
struct ForcedMod16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ModReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {fltTy, fltTy, strTy, intTy},
                                     {fltTy});
    };
  }
};

/// Placeholder for real*10 version of Modulo Intrinsic
struct ForcedModulo10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ModuloReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF80(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {fltTy, fltTy, strTy, intTy},
                                     {fltTy});
    };
  }
};

/// Placeholder for real*16 version of Modulo Intrinsic
struct ForcedModulo16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ModuloReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto fltTy = mlir::FloatType::getF128(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {fltTy, fltTy, strTy, intTy},
                                     {fltTy});
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

/// Generate call to Exponent intrinsic runtime routine.
mlir::Value fir::runtime::genExponent(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type resultType,
                                      mlir::Value x) {
  mlir::func::FuncOp func;
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
    fir::intrinsicTypeTODO(builder, fltTy, loc, "EXPONENT");

  auto funcTy = func.getFunctionType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Fraction intrinsic runtime routine.
mlir::Value fir::runtime::genFraction(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value x) {
  mlir::func::FuncOp func;
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
    fir::intrinsicTypeTODO(builder, fltTy, loc, "FRACTION");

  auto funcTy = func.getFunctionType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Mod intrinsic runtime routine.
mlir::Value fir::runtime::genMod(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value a, mlir::Value p) {
  mlir::func::FuncOp func;
  mlir::Type fltTy = a.getType();

  if (fltTy != p.getType())
    fir::emitFatalError(loc, "arguments type mismatch in MOD");

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ModReal4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ModReal8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedMod10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedMod16>(loc, builder);
  else
    fir::intrinsicTypeTODO(builder, fltTy, loc, "MOD");

  auto funcTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, funcTy, a, p,
                                            sourceFile, sourceLine);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Modulo intrinsic runtime routine.
mlir::Value fir::runtime::genModulo(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value a,
                                    mlir::Value p) {
  mlir::func::FuncOp func;
  mlir::Type fltTy = a.getType();

  if (fltTy != p.getType())
    fir::emitFatalError(loc, "arguments type mismatch in MOD");

  // MODULO is lowered into math operations in intrinsics lowering,
  // so genModulo() should only be used for F128 data type now.
  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ModuloReal4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ModuloReal8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedModulo10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedModulo16>(loc, builder);
  else
    fir::intrinsicTypeTODO(builder, fltTy, loc, "MODULO");

  auto funcTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, funcTy, a, p,
                                            sourceFile, sourceLine);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Nearest intrinsic or a "Next" intrinsic module procedure.
mlir::Value fir::runtime::genNearest(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value x,
                                     mlir::Value valueUp) {
  mlir::func::FuncOp func;
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
    fir::intrinsicTypeTODO(builder, fltTy, loc, "NEAREST");

  auto funcTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcTy, x, valueUp);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to RRSpacing intrinsic runtime routine.
mlir::Value fir::runtime::genRRSpacing(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value x) {
  mlir::func::FuncOp func;
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
    fir::intrinsicTypeTODO(builder, fltTy, loc, "RRSPACING");

  auto funcTy = func.getFunctionType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to ErfcScaled intrinsic runtime routine.
mlir::Value fir::runtime::genErfcScaled(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value x) {
  mlir::func::FuncOp func;
  mlir::Type fltTy = x.getType();

  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ErfcScaled4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ErfcScaled8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedErfcScaled10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedErfcScaled16>(loc, builder);
  else
    fir::intrinsicTypeTODO(builder, fltTy, loc, "ERFC_SCALED");

  auto funcTy = func.getFunctionType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Scale intrinsic runtime routine.
mlir::Value fir::runtime::genScale(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value x,
                                   mlir::Value i) {
  mlir::func::FuncOp func;
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
    fir::intrinsicTypeTODO(builder, fltTy, loc, "SCALE");

  auto funcTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcTy, x, i);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Selected_char_kind intrinsic runtime routine.
mlir::Value fir::runtime::genSelectedCharKind(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value name,
                                              mlir::Value length) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(SelectedCharKind)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(1));
  if (!fir::isa_ref_type(name.getType()))
    fir::emitFatalError(loc, "argument address for runtime not found");

  auto args = fir::runtime::createArguments(builder, loc, fTy, sourceFile,
                                            sourceLine, name, length);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Selected_int_kind intrinsic runtime routine.
mlir::Value fir::runtime::genSelectedIntKind(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value x) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(SelectedIntKind)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(1));
  if (!fir::isa_ref_type(x.getType()))
    fir::emitFatalError(loc, "argument address for runtime not found");
  mlir::Type eleTy = fir::unwrapRefType(x.getType());
  mlir::Value xKind = builder.createIntegerConstant(
      loc, fTy.getInput(3), eleTy.getIntOrFloatBitWidth() / 8);
  auto args = fir::runtime::createArguments(builder, loc, fTy, sourceFile,
                                            sourceLine, x, xKind);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Selected_logical_kind intrinsic runtime routine.
mlir::Value fir::runtime::genSelectedLogicalKind(fir::FirOpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::Value x) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(SelectedLogicalKind)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(1));
  if (!fir::isa_ref_type(x.getType()))
    fir::emitFatalError(loc, "argument address for runtime not found");
  mlir::Type eleTy = fir::unwrapRefType(x.getType());
  mlir::Value xKind = builder.createIntegerConstant(
      loc, fTy.getInput(3), eleTy.getIntOrFloatBitWidth() / 8);
  auto args = fir::runtime::createArguments(builder, loc, fTy, sourceFile,
                                            sourceLine, x, xKind);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Selected_real_kind intrinsic runtime routine.
mlir::Value fir::runtime::genSelectedRealKind(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value precision,
                                              mlir::Value range,
                                              mlir::Value radix) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(SelectedRealKind)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto getArgKinds = [&](mlir::Value arg, int argKindIndex) -> mlir::Value {
    if (fir::isa_ref_type(arg.getType())) {
      mlir::Type eleTy = fir::unwrapRefType(arg.getType());
      return builder.createIntegerConstant(loc, fTy.getInput(argKindIndex),
                                           eleTy.getIntOrFloatBitWidth() / 8);
    } else {
      return builder.createIntegerConstant(loc, fTy.getInput(argKindIndex), 0);
    }
  };

  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(1));
  mlir::Value pKind = getArgKinds(precision, 3);
  mlir::Value rKind = getArgKinds(range, 5);
  mlir::Value dKind = getArgKinds(radix, 7);
  auto args = fir::runtime::createArguments(builder, loc, fTy, sourceFile,
                                            sourceLine, precision, pKind, range,
                                            rKind, radix, dKind);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Set_exponent intrinsic runtime routine.
mlir::Value fir::runtime::genSetExponent(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value x,
                                         mlir::Value i) {
  mlir::func::FuncOp func;
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
    fir::intrinsicTypeTODO(builder, fltTy, loc, "SET_EXPONENT");

  auto funcTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcTy, x, i);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Spacing intrinsic runtime routine.
mlir::Value fir::runtime::genSpacing(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value x) {
  mlir::func::FuncOp func;
  mlir::Type fltTy = x.getType();
  // TODO: for f16/bf16, there are better alternatives that do not require
  // casting the argument (resp. result) to (resp. from) f32, but this requires
  // knowing that the target runtime has been compiled with std::float16_t or
  // std::bfloat16_t support, which is not an information available here for
  // now.
  if (fltTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Spacing4)>(loc, builder);
  else if (fltTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Spacing8)>(loc, builder);
  else if (fltTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedSpacing10>(loc, builder);
  else if (fltTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedSpacing16>(loc, builder);
  else if (fltTy.isF16())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Spacing2By4)>(loc, builder);
  else if (fltTy.isBF16())
    func = fir::runtime::getRuntimeFunc<mkRTKey(Spacing3By4)>(loc, builder);
  else
    fir::intrinsicTypeTODO(builder, fltTy, loc, "SPACING");

  auto funcTy = func.getFunctionType();
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, funcTy.getInput(0), x)};

  mlir::Value res = builder.create<fir::CallOp>(loc, func, args).getResult(0);
  return builder.createConvert(loc, fltTy, res);
}
