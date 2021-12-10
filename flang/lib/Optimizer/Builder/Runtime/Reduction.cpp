//===-- Reduction.cpp -- generate reduction intrinsics runtime calls- -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/reduction.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Placeholder for real*10 version of Maxval Intrinsic
struct ForcedMaxvalReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MaxvalReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Maxval Intrinsic
struct ForcedMaxvalReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MaxvalReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Maxval Intrinsic
struct ForcedMaxvalInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MaxvalInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*10 version of Minval Intrinsic
struct ForcedMinvalReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MinvalReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Minval Intrinsic
struct ForcedMinvalReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MinvalReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Minval Intrinsic
struct ForcedMinvalInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MinvalInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*10 version of Product Intrinsic
struct ForcedProductReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ProductReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Product Intrinsic
struct ForcedProductReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(ProductReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Product Intrinsic
struct ForcedProductInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(ProductInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for complex(10) version of Product Intrinsic
struct ForcedProductComplex10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppProductComplex10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::FloatType::getF80(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for complex(16) version of Product Intrinsic
struct ForcedProductComplex16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppProductComplex16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::FloatType::getF128(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for real*10 version of DotProduct Intrinsic
struct ForcedDotProductReal10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for real*16 version of DotProduct Intrinsic
struct ForcedDotProductReal16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for complex(10) version of DotProduct Intrinsic
struct ForcedDotProductComplex10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppDotProductComplex10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::FloatType::getF80(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(ctx, {resTy, boxTy, boxTy, strTy, intTy},
                                     {});
    };
  }
};

/// Placeholder for complex(16) version of DotProduct Intrinsic
struct ForcedDotProductComplex16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppDotProductComplex16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::FloatType::getF128(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(ctx, {resTy, boxTy, boxTy, strTy, intTy},
                                     {});
    };
  }
};

/// Placeholder for integer*16 version of DotProduct Intrinsic
struct ForcedDotProductInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(DotProductInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, boxTy, strTy, intTy}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Sum Intrinsic
struct ForcedSumReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumReal10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for real*16 version of Sum Intrinsic
struct ForcedSumReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumReal16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for integer*16 version of Sum Intrinsic
struct ForcedSumInteger16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(SumInteger16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      return mlir::FunctionType::get(ctx, {boxTy, strTy, intTy, intTy, boxTy},
                                     {ty});
    };
  }
};

/// Placeholder for complex(10) version of Sum Intrinsic
struct ForcedSumComplex10 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppSumComplex10));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::FloatType::getF80(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Placeholder for complex(16) version of Sum Intrinsic
struct ForcedSumComplex16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(CppSumComplex16));
  static constexpr fir::runtime::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::ComplexType::get(mlir::FloatType::getF128(ctx));
      auto boxTy =
          fir::runtime::getModel<const Fortran::runtime::Descriptor &>()(ctx);
      auto strTy = fir::ReferenceType::get(mlir::IntegerType::get(ctx, 8));
      auto intTy = mlir::IntegerType::get(ctx, 8 * sizeof(int));
      auto resTy = fir::ReferenceType::get(ty);
      return mlir::FunctionType::get(
          ctx, {resTy, boxTy, strTy, intTy, intTy, boxTy}, {});
    };
  }
};

/// Generate call to specialized runtime function that takes a mask and
/// dim argument. The All, Any, and Count intrinsics use this pattern.
template <typename FN>
mlir::Value genSpecial2Args(FN func, fir::FirOpBuilder &builder,
                            mlir::Location loc, mlir::Value maskBox,
                            mlir::Value dim) {
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, maskBox,
                                            sourceFile, sourceLine, dim);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate calls to reduction intrinsics such as All and Any.
/// These are the descriptor based implementations that take two
/// arguments (mask, dim).
template <typename FN>
static void genReduction2Args(FN func, fir::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value maskBox, mlir::Value dim) {
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, maskBox, dim, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxval and Minval.
/// These take arguments such as (array, dim, mask).
template <typename FN>
static void genReduction3Args(FN func, fir::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value dim,
                              mlir::Value maskBox) {

  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox, dim,
                                    sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxloc and Minloc.
/// These take arguments such as (array, mask, kind, back).
template <typename FN>
static void genReduction4Args(FN func, fir::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value maskBox,
                              mlir::Value kind, mlir::Value back) {
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            arrayBox, kind, sourceFile,
                                            sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxloc and Minloc.
/// These take arguments such as (array, dim, mask, kind, back).
template <typename FN>
static void
genReduction5Args(FN func, fir::FirOpBuilder &builder, mlir::Location loc,
                  mlir::Value resultBox, mlir::Value arrayBox, mlir::Value dim,
                  mlir::Value maskBox, mlir::Value kind, mlir::Value back) {
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            arrayBox, kind, dim, sourceFile,
                                            sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `AllDim` runtime routine.
/// This calls the descriptor based runtime call implementation of the `all`
/// intrinsic.
void fir::runtime::genAllDescriptor(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value resultBox,
                                    mlir::Value maskBox, mlir::Value dim) {
  auto allFunc = fir::runtime::getRuntimeFunc<mkRTKey(AllDim)>(loc, builder);
  genReduction2Args(allFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to `AnyDim` runtime routine.
/// This calls the descriptor based runtime call implementation of the `any`
/// intrinsic.
void fir::runtime::genAnyDescriptor(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value resultBox,
                                    mlir::Value maskBox, mlir::Value dim) {
  auto anyFunc = fir::runtime::getRuntimeFunc<mkRTKey(AnyDim)>(loc, builder);
  genReduction2Args(anyFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to `All` intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value fir::runtime::genAll(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value maskBox, mlir::Value dim) {
  auto allFunc = fir::runtime::getRuntimeFunc<mkRTKey(All)>(loc, builder);
  return genSpecial2Args(allFunc, builder, loc, maskBox, dim);
}

/// Generate call to `Any` intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value fir::runtime::genAny(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value maskBox, mlir::Value dim) {
  auto anyFunc = fir::runtime::getRuntimeFunc<mkRTKey(Any)>(loc, builder);
  return genSpecial2Args(anyFunc, builder, loc, maskBox, dim);
}

/// Generate call to `Count` runtime routine. This routine is a specialized
/// version when mask is a rank one array or the dim argument is not
/// specified by the user.
mlir::Value fir::runtime::genCount(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value maskBox,
                                   mlir::Value dim) {
  auto countFunc = fir::runtime::getRuntimeFunc<mkRTKey(Count)>(loc, builder);
  return genSpecial2Args(countFunc, builder, loc, maskBox, dim);
}

/// Generate call to general `CountDim` runtime routine. This routine has a
/// descriptor result.
void fir::runtime::genCountDim(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value resultBox, mlir::Value maskBox,
                               mlir::Value dim, mlir::Value kind) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(CountDim)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, maskBox, dim, kind, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Maxloc` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void fir::runtime::genMaxloc(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value maskBox, mlir::Value kind,
                             mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Maxloc)>(loc, builder);
  genReduction4Args(func, builder, loc, resultBox, arrayBox, maskBox, kind,
                    back);
}

/// Generate call to `MaxlocDim` intrinsic runtime routine. This is the version
/// that takes a dim argument.
void fir::runtime::genMaxlocDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox,
                                mlir::Value kind, mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MaxlocDim)>(loc, builder);
  genReduction5Args(func, builder, loc, resultBox, arrayBox, dim, maskBox, kind,
                    back);
}

/// Generate call to `Maxval` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genMaxval(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value arrayBox,
                                    mlir::Value maskBox) {
  mlir::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedMaxvalReal10>(loc, builder);
  else if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedMaxvalReal16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func = fir::runtime::getRuntimeFunc<ForcedMaxvalInteger16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in Maxval lowering");

  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `MaxvalDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genMaxvalDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MaxvalDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `MaxvalCharacter` intrinsic runtime routine. This is the
/// version that handles character arrays of rank 1 and without a DIM argument.
void fir::runtime::genMaxvalChar(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value maskBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(MaxvalCharacter)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Minloc` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void fir::runtime::genMinloc(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value maskBox, mlir::Value kind,
                             mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Minloc)>(loc, builder);
  genReduction4Args(func, builder, loc, resultBox, arrayBox, maskBox, kind,
                    back);
}

/// Generate call to `MinlocDim` intrinsic runtime routine. This is the version
/// that takes a dim argument.
void fir::runtime::genMinlocDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox,
                                mlir::Value kind, mlir::Value back) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MinlocDim)>(loc, builder);
  genReduction5Args(func, builder, loc, resultBox, arrayBox, dim, maskBox, kind,
                    back);
}

/// Generate call to `MinvalDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genMinvalDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value resultBox, mlir::Value arrayBox,
                                mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `MinvalCharacter` intrinsic runtime routine. This is the
/// version that handles character arrays of rank 1 and without a DIM argument.
void fir::runtime::genMinvalChar(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value maskBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(MinvalCharacter)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, arrayBox, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to `Minval` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genMinval(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value arrayBox,
                                    mlir::Value maskBox) {
  mlir::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedMinvalReal10>(loc, builder);
  else if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedMinvalReal16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(MinvalInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func = fir::runtime::getRuntimeFunc<ForcedMinvalInteger16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in Minval lowering");

  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `ProductDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genProductDim(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value resultBox, mlir::Value arrayBox,
                                 mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(ProductDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `Product` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genProduct(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value arrayBox,
                                     mlir::Value maskBox,
                                     mlir::Value resultBox) {
  mlir::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ProductReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(ProductReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedProductReal10>(loc, builder);
  else if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedProductReal16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(ProductInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(ProductInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(ProductInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(ProductInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func = fir::runtime::getRuntimeFunc<ForcedProductInteger16>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 4))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(CppProductComplex4)>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 8))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(CppProductComplex8)>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 10))
    func = fir::runtime::getRuntimeFunc<ForcedProductComplex10>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 16))
    func = fir::runtime::getRuntimeFunc<ForcedProductComplex16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in Product lowering");

  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  if (fir::isa_complex(eleTy)) {
    auto sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
    auto args =
        fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                      sourceFile, sourceLine, dim, maskBox);
    builder.create<fir::CallOp>(loc, func, args);
    return resultBox;
  }

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to `DotProduct` intrinsic runtime routine.
mlir::Value fir::runtime::genDotProduct(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value vectorABox,
                                        mlir::Value vectorBBox,
                                        mlir::Value resultBox) {
  mlir::FuncOp func;
  auto ty = vectorABox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();

  if (eleTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(DotProductReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(DotProductReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedDotProductReal10>(loc, builder);
  else if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedDotProductReal16>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 4))
    func = fir::runtime::getRuntimeFunc<mkRTKey(CppDotProductComplex4)>(
        loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 8))
    func = fir::runtime::getRuntimeFunc<mkRTKey(CppDotProductComplex8)>(
        loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 10))
    func =
        fir::runtime::getRuntimeFunc<ForcedDotProductComplex10>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 16))
    func =
        fir::runtime::getRuntimeFunc<ForcedDotProductComplex16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(DotProductInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(DotProductInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(DotProductInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(DotProductInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func =
        fir::runtime::getRuntimeFunc<ForcedDotProductInteger16>(loc, builder);
  else if (eleTy.isa<fir::LogicalType>())
    func =
        fir::runtime::getRuntimeFunc<mkRTKey(DotProductLogical)>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in DotProduct lowering");

  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);

  if (fir::isa_complex(eleTy)) {
    auto sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
    auto args =
        fir::runtime::createArguments(builder, loc, fTy, resultBox, vectorABox,
                                      vectorBBox, sourceFile, sourceLine);
    builder.create<fir::CallOp>(loc, func, args);
    return resultBox;
  }

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, vectorABox,
                                            vectorBBox, sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
/// Generate call to `SumDim` intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void fir::runtime::genSumDim(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value arrayBox,
                             mlir::Value dim, mlir::Value maskBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(SumDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to `Sum` intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value fir::runtime::genSum(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value arrayBox, mlir::Value maskBox,
                                 mlir::Value resultBox) {
  mlir::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = fir::runtime::getRuntimeFunc<mkRTKey(SumReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = fir::runtime::getRuntimeFunc<mkRTKey(SumReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = fir::runtime::getRuntimeFunc<ForcedSumReal10>(loc, builder);
  else if (eleTy.isF128())
    func = fir::runtime::getRuntimeFunc<ForcedSumReal16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(SumInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(SumInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(SumInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func = fir::runtime::getRuntimeFunc<mkRTKey(SumInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func = fir::runtime::getRuntimeFunc<ForcedSumInteger16>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 4))
    func = fir::runtime::getRuntimeFunc<mkRTKey(CppSumComplex4)>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 8))
    func = fir::runtime::getRuntimeFunc<mkRTKey(CppSumComplex8)>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 10))
    func = fir::runtime::getRuntimeFunc<ForcedSumComplex10>(loc, builder);
  else if (eleTy == fir::ComplexType::get(builder.getContext(), 16))
    func = fir::runtime::getRuntimeFunc<ForcedSumComplex16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in Sum lowering");

  auto fTy = func.getType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  if (fir::isa_complex(eleTy)) {
    auto sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
    auto args =
        fir::runtime::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                      sourceFile, sourceLine, dim, maskBox);
    builder.create<fir::CallOp>(loc, func, args);
    return resultBox;
  }

  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
