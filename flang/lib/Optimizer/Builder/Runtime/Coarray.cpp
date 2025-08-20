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

/// Generate Call to runtime prif_num_images
mlir::Value fir::runtime::getNumImages(fir::FirOpBuilder &builder,
                                       mlir::Location loc) {
  mlir::Value result = builder.createTemporary(loc, builder.getI32Type());
  mlir::FunctionType ftype =
      PRIF_FUNCTYPE(builder.getRefType(builder.getI32Type()));
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("num_images"), ftype);
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, ftype, result);
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}

/// Generate Call to runtime prif_num_images_with_{team|team_number}
mlir::Value fir::runtime::getNumImagesWithTeam(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Value team) {
  bool isTeamNumber = fir::unwrapPassByRefType(team.getType()).isInteger();
  std::string numImagesName = isTeamNumber
                                  ? PRIFNAME_SUB("num_images_with_team_number")
                                  : PRIFNAME_SUB("num_images_with_team");

  mlir::Value result = builder.createTemporary(loc, builder.getI32Type());
  mlir::Type refTy = builder.getRefType(builder.getI32Type());
  mlir::FunctionType ftype =
      isTeamNumber
          ? PRIF_FUNCTYPE(builder.getRefType(builder.getI64Type()), refTy)
          : PRIF_FUNCTYPE(fir::BoxType::get(builder.getNoneType()), refTy);
  mlir::func::FuncOp funcOp = builder.createFunction(loc, numImagesName, ftype);

  if (!isTeamNumber)
    team = builder.createBox(loc, team);
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, ftype, team, result);
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}

/// Generate Call to runtime prif_this_image_no_coarray
mlir::Value fir::runtime::getThisImage(fir::FirOpBuilder &builder,
                                       mlir::Location loc, mlir::Value team) {
  mlir::Type refTy = builder.getRefType(builder.getI32Type());
  mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
  mlir::FunctionType ftype = PRIF_FUNCTYPE(boxTy, refTy);
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, PRIFNAME_SUB("this_image_no_coarray"), ftype);

  mlir::Value result = builder.createTemporary(loc, builder.getI32Type());
  mlir::Value teamArg =
      !team ? builder.create<fir::AbsentOp>(loc, boxTy) : team;
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, ftype, teamArg, result);
  builder.create<fir::CallOp>(loc, funcOp, args);
  return builder.create<fir::LoadOp>(loc, result);
}
