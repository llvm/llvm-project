//===-- Coarray.cpp -- runtime API for coarray intrinsics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Coarray.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;
using namespace Fortran::semantics;

/// Generate Call to runtime prif_init
mlir::Value fir::runtime::genInitCoarray(fir::FirOpBuilder &builder,
                                         mlir::Location loc) {
  mlir::Type i32Ty = builder.getI32Type();
  mlir::Value result = builder.createTemporary(loc, i32Ty);
  mlir::FunctionType ftype = PRIF_FUNCTYPE(builder.getRefType(i32Ty));
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("init"), ftype);
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, ftype, result);
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}
