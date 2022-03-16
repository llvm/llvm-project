//===-- Ragged.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Ragged.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/ragged.h"

using namespace Fortran::runtime;

void fir::runtime::genRaggedArrayAllocate(mlir::Location loc,
                                          fir::FirOpBuilder &builder,
                                          mlir::Value header, bool asHeaders,
                                          mlir::Value eleSize,
                                          mlir::ValueRange extents) {
  auto i32Ty = builder.getIntegerType(32);
  auto rank = extents.size();
  auto i64Ty = builder.getIntegerType(64);
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(RaggedArrayAllocate)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto i1Ty = builder.getIntegerType(1);
  fir::SequenceType::Shape shape = {
      static_cast<fir::SequenceType::Extent>(rank)};
  auto extentTy = fir::SequenceType::get(shape, i64Ty);
  auto refTy = fir::ReferenceType::get(i64Ty);
  // Position of the bufferPointer in the header struct.
  auto one = builder.createIntegerConstant(loc, i32Ty, 1);
  auto eleTy = fir::unwrapSequenceType(fir::unwrapRefType(header.getType()));
  auto ptrTy = builder.getRefType(eleTy.cast<mlir::TupleType>().getType(1));
  auto ptr = builder.create<fir::CoordinateOp>(loc, ptrTy, header, one);
  auto heap = builder.create<fir::LoadOp>(loc, ptr);
  auto cmp = builder.genIsNull(loc, heap);
  builder.genIfThen(loc, cmp)
      .genThen([&]() {
        auto asHeadersVal = builder.createIntegerConstant(loc, i1Ty, asHeaders);
        auto rankVal = builder.createIntegerConstant(loc, i64Ty, rank);
        auto buff = builder.create<fir::AllocMemOp>(loc, extentTy);
        // Convert all the extents to i64 and pack them in a buffer on the heap.
        for (auto i : llvm::enumerate(extents)) {
          auto offset = builder.createIntegerConstant(loc, i32Ty, i.index());
          auto addr =
              builder.create<fir::CoordinateOp>(loc, refTy, buff, offset);
          auto castVal = builder.createConvert(loc, i64Ty, i.value());
          builder.create<fir::StoreOp>(loc, castVal, addr);
        }
        auto args = fir::runtime::createArguments(
            builder, loc, fTy, header, asHeadersVal, rankVal, eleSize, buff);
        builder.create<fir::CallOp>(loc, func, args);
      })
      .end();
}

void fir::runtime::genRaggedArrayDeallocate(mlir::Location loc,
                                            fir::FirOpBuilder &builder,
                                            mlir::Value header) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(RaggedArrayDeallocate)>(
      loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, header);
  builder.create<fir::CallOp>(loc, func, args);
}
