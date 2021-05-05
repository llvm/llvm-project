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

/// Placeholder for real*10 version of Maxval Intrinsic
struct ForcedMaxvalReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MaxvalReal10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Maxval Intrinsic
struct ForcedMaxvalReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MaxvalReal16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for integer*16 version of Maxval Intrinsic
struct ForcedMaxvalInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MaxvalInteger16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*10 version of Minval Intrinsic
struct ForcedMinvalReal10 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MinvalReal10));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF80(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for real*16 version of Minval Intrinsic
struct ForcedMinvalReal16 {
  static constexpr const char *name = ExpandAndQuoteKey(RTNAME(MinvalReal16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::FloatType::getF128(ctx);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Placeholder for integer*16 version of Minval Intrinsic
struct ForcedMinvalInteger16 {
  static constexpr const char *name =
      ExpandAndQuoteKey(RTNAME(MinvalInteger16));
  static constexpr Fortran::lower::FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctx) {
      auto ty = mlir::IntegerType::get(ctx, 128);
      return mlir::FunctionType::get(ctx, {ty}, {ty});
    };
  }
};

/// Generate calls to reduction intrinsics such as All and Any.
/// These are the descriptor based implementations that take two
/// arguments (mask, dim).
template <typename FN>
static void genReduction2Args(FN func, Fortran::lower::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value maskBox, mlir::Value dim) {
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, maskBox, dim, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxval and Minval.
/// These take arguments such as (array, dim, mask).
template <typename FN>
static void genReduction3Args(FN func, Fortran::lower::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value dim,
                              mlir::Value maskBox) {

  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args =
      Fortran::lower::createArguments(builder, loc, fTy, resultBox, arrayBox,
                                      dim, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxloc and Minloc.
/// These take arguments such as (array, mask, kind, back).
template <typename FN>
static void genReduction4Args(FN func, Fortran::lower::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value maskBox,
                              mlir::Value kind, mlir::Value back) {
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(4));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, resultBox,
                                              arrayBox, kind, sourceFile,
                                              sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate calls to reduction intrinsics such as Maxloc and Minloc.
/// These take arguments such as (array, dim, mask, kind, back).
template <typename FN>
static void genReduction5Args(FN func, Fortran::lower::FirOpBuilder &builder,
                              mlir::Location loc, mlir::Value resultBox,
                              mlir::Value arrayBox, mlir::Value dim,
                              mlir::Value maskBox, mlir::Value kind,
                              mlir::Value back) {
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(5));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, resultBox,
                                              arrayBox, kind, dim, sourceFile,
                                              sourceLine, maskBox, back);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to all runtime routine.
/// This calls the descriptor based runtime call implementation of the all
/// intrinsic.
void Fortran::lower::genAllDescriptor(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value resultBox,
                                      mlir::Value maskBox, mlir::Value dim) {
  auto allFunc = Fortran::lower::getRuntimeFunc<mkRTKey(AllDim)>(loc, builder);
  genReduction2Args(allFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to any runtime routine.
/// This calls the descriptor based runtime call implementation of the any
/// intrinsic.
void Fortran::lower::genAnyDescriptor(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value resultBox,
                                      mlir::Value maskBox, mlir::Value dim) {
  auto anyFunc = Fortran::lower::getRuntimeFunc<mkRTKey(AnyDim)>(loc, builder);
  genReduction2Args(anyFunc, builder, loc, resultBox, maskBox, dim);
}

/// Generate call to All intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value Fortran::lower::genAll(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value maskBox,
                                   mlir::Value dim) {
  auto allFunc = Fortran::lower::getRuntimeFunc<mkRTKey(All)>(loc, builder);
  auto fTy = allFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, maskBox,
                                              sourceFile, sourceLine, dim);
  return builder.create<fir::CallOp>(loc, allFunc, args).getResult(0);
}

/// Generate call to Any intrinsic runtime routine. This routine is
/// specialized for mask arguments with rank == 1.
mlir::Value Fortran::lower::genAny(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value maskBox,
                                   mlir::Value dim) {
  auto anyFunc = Fortran::lower::getRuntimeFunc<mkRTKey(Any)>(loc, builder);
  auto fTy = anyFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = Fortran::lower::createArguments(builder, loc, fTy, maskBox,
                                              sourceFile, sourceLine, dim);
  return builder.create<fir::CallOp>(loc, anyFunc, args).getResult(0);
}

/// Generate call to Maxloc intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void Fortran::lower::genMaxloc(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value arrayBox, mlir::Value maskBox,
                               mlir::Value kind, mlir::Value back) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Maxloc)>(loc, builder);
  genReduction4Args(func, builder, loc, resultBox, arrayBox, maskBox, kind,
                    back);
}

/// Generate call to Maxloc intrinsic runtime routine. This is the version
/// that takes a dim argument.
void Fortran::lower::genMaxlocDim(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value resultBox,
                                  mlir::Value arrayBox, mlir::Value dim,
                                  mlir::Value maskBox, mlir::Value kind,
                                  mlir::Value back) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(MaxlocDim)>(loc, builder);
  genReduction5Args(func, builder, loc, resultBox, arrayBox, dim, maskBox, kind,
                    back);
}

/// Generate call to Maxval intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value Fortran::lower::genMaxval(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value arrayBox,
                                      mlir::Value maskBox) {
  mlir::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedMaxvalReal10>(loc, builder);
  else if (eleTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedMaxvalReal16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func = Fortran::lower::getRuntimeFunc<ForcedMaxvalInteger16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in Maxval lowering");

  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);
  builder.create<fir::CallOp>(loc, func, args);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to Maxval intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void Fortran::lower::genMaxvalDim(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value resultBox,
                                  mlir::Value arrayBox, mlir::Value dim,
                                  mlir::Value maskBox) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to Maxval intrinsic runtime routine. This is the version
/// that handles character arrays of rank 1 and without a DIM argument.
void Fortran::lower::genMaxvalChar(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value resultBox,
                                   mlir::Value arrayBox, mlir::Value maskBox) {
  auto func =
      Fortran::lower::getRuntimeFunc<mkRTKey(MaxvalCharacter)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, arrayBox, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Minloc intrinsic runtime routine. This is the version
/// that does not take a dim argument.
void Fortran::lower::genMinloc(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value arrayBox, mlir::Value maskBox,
                               mlir::Value kind, mlir::Value back) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(Minloc)>(loc, builder);
  genReduction4Args(func, builder, loc, resultBox, arrayBox, maskBox, kind,
                    back);
}

/// Generate call to Minloc intrinsic runtime routine. This is the version
/// that takes a dim argument.
void Fortran::lower::genMinlocDim(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value resultBox,
                                  mlir::Value arrayBox, mlir::Value dim,
                                  mlir::Value maskBox, mlir::Value kind,
                                  mlir::Value back) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(MinlocDim)>(loc, builder);
  genReduction5Args(func, builder, loc, resultBox, arrayBox, dim, maskBox, kind,
                    back);
}

/// Generate call to Minval intrinsic runtime routine. This is the version
/// that handles any rank array with the dim argument specified.
void Fortran::lower::genMinvalDim(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value resultBox,
                                  mlir::Value arrayBox, mlir::Value dim,
                                  mlir::Value maskBox) {
  auto func = Fortran::lower::getRuntimeFunc<mkRTKey(MinvalDim)>(loc, builder);
  genReduction3Args(func, builder, loc, resultBox, arrayBox, dim, maskBox);
}

/// Generate call to Minval intrinsic runtime routine. This is the version
/// that handles character arrays of rank 1 and without a DIM argument.
void Fortran::lower::genMinvalChar(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value resultBox,
                                   mlir::Value arrayBox, mlir::Value maskBox) {
  auto func =
      Fortran::lower::getRuntimeFunc<mkRTKey(MinvalCharacter)>(loc, builder);
  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, arrayBox, sourceFile, sourceLine, maskBox);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Generate call to Minval intrinsic runtime routine. This is the version
/// that does not take a dim argument.
mlir::Value Fortran::lower::genMinval(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value arrayBox,
                                      mlir::Value maskBox) {
  mlir::FuncOp func;
  auto ty = arrayBox.getType();
  auto arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
  auto dim = builder.createIntegerConstant(loc, builder.getIndexType(), 0);

  if (eleTy.isF32())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(MinvalReal4)>(loc, builder);
  else if (eleTy.isF64())
    func = Fortran::lower::getRuntimeFunc<mkRTKey(MinvalReal8)>(loc, builder);
  else if (eleTy.isF80())
    func = Fortran::lower::getRuntimeFunc<ForcedMinvalReal10>(loc, builder);
  else if (eleTy.isF128())
    func = Fortran::lower::getRuntimeFunc<ForcedMinvalReal16>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(1)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MinvalInteger1)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(2)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MinvalInteger2)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(4)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MinvalInteger4)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(8)))
    func =
        Fortran::lower::getRuntimeFunc<mkRTKey(MinvalInteger8)>(loc, builder);
  else if (eleTy ==
           builder.getIntegerType(builder.getKindMap().getIntegerBitsize(16)))
    func = Fortran::lower::getRuntimeFunc<ForcedMinvalInteger16>(loc, builder);
  else
    fir::emitFatalError(loc, "invalid type in Minval lowering");

  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, arrayBox, sourceFile, sourceLine, dim, maskBox);
  builder.create<fir::CallOp>(loc, func, args);

  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
