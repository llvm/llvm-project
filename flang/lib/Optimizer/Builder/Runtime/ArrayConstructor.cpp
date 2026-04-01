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
#include "flang/Runtime/array-constructor-consts.h"

using namespace Fortran::runtime;

namespace fir::runtime {
template <>
constexpr TypeBuilderFunc
getModel<Fortran::runtime::ArrayConstructorVector &>() {
  return getModel<void *>();
}
} // namespace fir::runtime

aiir::Value fir::runtime::genInitArrayConstructorVector(
    aiir::Location loc, fir::FirOpBuilder &builder, aiir::Value toBox,
    aiir::Value useValueLengthParameters) {
  // Allocate storage for the runtime cookie for the array constructor vector.
  // Use pessimistic values for size and alignment that are valid for all
  // supported targets. Whether the actual ArrayConstructorVector object fits
  // into the available MaxArrayConstructorVectorSizeInBytes is verified when
  // building clang-rt.
  std::size_t arrayVectorStructBitSize =
      MaxArrayConstructorVectorSizeInBytes * 8;
  std::size_t alignLike = MaxArrayConstructorVectorAlignInBytes * 8;
  fir::SequenceType::Extent numElem =
      (arrayVectorStructBitSize + alignLike - 1) / alignLike;
  aiir::Type intType = builder.getIntegerType(alignLike);
  aiir::Type seqType = fir::SequenceType::get({numElem}, intType);
  aiir::Value cookie =
      builder.createTemporary(loc, seqType, ".rt.arrayctor.vector");

  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(InitArrayConstructorVector)>(
          loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  cookie = builder.createConvert(loc, funcType.getInput(0), cookie);
  aiir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  aiir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, funcType.getInput(4));
  auto args = fir::runtime::createArguments(builder, loc, funcType, cookie,
                                            toBox, useValueLengthParameters,
                                            sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
  return cookie;
}

void fir::runtime::genPushArrayConstructorValue(
    aiir::Location loc, fir::FirOpBuilder &builder,
    aiir::Value arrayConstructorVector, aiir::Value fromBox) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushArrayConstructorValue)>(loc,
                                                                       builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, funcType,
                                            arrayConstructorVector, fromBox);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genPushArrayConstructorSimpleScalar(
    aiir::Location loc, fir::FirOpBuilder &builder,
    aiir::Value arrayConstructorVector, aiir::Value fromAddress) {
  aiir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(PushArrayConstructorSimpleScalar)>(
          loc, builder);
  aiir::FunctionType funcType = func.getFunctionType();
  auto args = fir::runtime::createArguments(
      builder, loc, funcType, arrayConstructorVector, fromAddress);
  fir::CallOp::create(builder, loc, func, args);
}
