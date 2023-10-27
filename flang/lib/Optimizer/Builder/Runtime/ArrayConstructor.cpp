//===- ArrayConstructor.cpp - array constructor runtime API calls ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/ArrayConstructor.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/array-constructor.h"

using namespace Fortran::runtime;

namespace fir::runtime {
template <>
constexpr TypeBuilderFunc
getModel<Fortran::runtime::ArrayConstructorVector &>() {
  return getModel<void *>();
}
} // namespace fir::runtime

mlir::Value fir::runtime::genInitArrayConstructorVector(
    mlir::Location loc, fir::FirOpBuilder &builder, mlir::Value toBox,
    mlir::Value useValueLengthParameters) {
  // Allocate storage for the runtime cookie for the array constructor vector.
  // Use the "host" size and alignment, but double them to be safe regardless of
  // the target. The "cookieSize" argument is used to validate this wild
  // assumption until runtime interfaces are improved.
  std::size_t arrayVectorStructBitSize =
      2 * sizeof(Fortran::runtime::ArrayConstructorVector) * 8;
  std::size_t alignLike = alignof(Fortran::runtime::ArrayConstructorVector) * 8;
  fir::SequenceType::Extent numElem =
      (arrayVectorStructBitSize + alignLike - 1) / alignLike;
  mlir::Type intType = builder.getIntegerType(alignLike);
  mlir::Type seqType = fir::SequenceType::get({numElem}, intType);
  mlir::Value cookie =
      builder.createTemporary(loc, seqType, ".rt.arrayctor.vector");

  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(InitArrayConstructorVector)>(
          loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  cookie = builder.createConvert(loc, funcType.getInput(0), cookie);
  mlir::Value cookieSize = builder.createIntegerConstant(
      loc, funcType.getInput(3), numElem * alignLike / 8);
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcType.getInput(5));
  auto args = fir::runtime::createArguments(builder, loc, funcType, cookie,
                                            toBox, useValueLengthParameters,
                                            cookieSize, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
  return cookie;
}

void fir::runtime::genPushArrayConstructorValue(
    mlir::Location loc, fir::FirOpBuilder &builder,
    mlir::Value arrayConstructorVector, mlir::Value fromBox) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushArrayConstructorValue)>(loc,
                                                                       builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType,
                                            arrayConstructorVector, fromBox);
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::genPushArrayConstructorSimpleScalar(
    mlir::Location loc, fir::FirOpBuilder &builder,
    mlir::Value arrayConstructorVector, mlir::Value fromAddress) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushArrayConstructorSimpleScalar)>(
          loc, builder);
  mlir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(
      builder, loc, funcType, arrayConstructorVector, fromAddress);
  builder.create<fir::CallOp>(loc, func, args);
}
