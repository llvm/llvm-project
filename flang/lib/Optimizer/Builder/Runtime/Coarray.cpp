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

// Most PRIF functions take `errmsg` and `errmsg_alloc` as two optional
// arguments of intent (out). One is allocatable, the other is not.
// It is the responsibility of the compiler to ensure that the appropriate
// optional argument is passed, and at most one must be provided in a given
// call.
// Depending on the type of `errmsg`, this function will return the pair
// corresponding to (`errmsg`, `errmsg_alloc`).
static std::pair<mlir::Value, mlir::Value>
genErrmsgPRIF(fir::FirOpBuilder &builder, mlir::Location loc,
              mlir::Value errmsg) {
  bool isAllocatableErrmsg = fir::isAllocatableType(errmsg.getType());

  mlir::Value absent = fir::AbsentOp::create(builder, loc, PRIF_ERRMSG_TYPE);
  mlir::Value errMsg = isAllocatableErrmsg ? absent : errmsg;
  mlir::Value errMsgAlloc = isAllocatableErrmsg ? errmsg : absent;
  return {errMsg, errMsgAlloc};
}

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
  fir::CallOp::create(builder, loc, funcOp, args);
  return fir::LoadOp::create(builder, loc, result);
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
  fir::CallOp::create(builder, loc, funcOp, args);
  return fir::LoadOp::create(builder, loc, result);
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
  fir::CallOp::create(builder, loc, funcOp, args);
  return fir::LoadOp::create(builder, loc, result);
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
      !team ? fir::AbsentOp::create(builder, loc, boxTy) : team;
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, ftype, teamArg, result);
  fir::CallOp::create(builder, loc, funcOp, args);
  return fir::LoadOp::create(builder, loc, result);
}

/// Generate call to collective subroutines except co_reduce
/// A must be lowered as a box
void genCollectiveSubroutine(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value A, mlir::Value rootImage,
                             mlir::Value stat, mlir::Value errmsg,
                             std::string coName) {
  mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
  mlir::FunctionType ftype =
      PRIF_FUNCTYPE(boxTy, builder.getRefType(builder.getI32Type()),
                    PRIF_STAT_TYPE, PRIF_ERRMSG_TYPE, PRIF_ERRMSG_TYPE);
  mlir::func::FuncOp funcOp = builder.createFunction(loc, coName, ftype);

  auto [errmsgArg, errmsgAllocArg] = genErrmsgPRIF(builder, loc, errmsg);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, ftype, A, rootImage, stat, errmsgArg, errmsgAllocArg);
  fir::CallOp::create(builder, loc, funcOp, args);
}

/// Generate call to runtime subroutine prif_co_broadcast
void fir::runtime::genCoBroadcast(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value A,
                                  mlir::Value sourceImage, mlir::Value stat,
                                  mlir::Value errmsg) {
  genCollectiveSubroutine(builder, loc, A, sourceImage, stat, errmsg,
                          PRIFNAME_SUB("co_broadcast"));
}

/// Generate call to runtime subroutine prif_co_max or prif_co_max_character
void fir::runtime::genCoMax(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value A, mlir::Value resultImage,
                            mlir::Value stat, mlir::Value errmsg) {
  mlir::Type argTy =
      fir::unwrapSequenceType(fir::unwrapPassByRefType(A.getType()));
  if (mlir::isa<fir::CharacterType>(argTy))
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_max_character"));
  else
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_max"));
}

/// Generate call to runtime subroutine prif_co_min or prif_co_min_character
void fir::runtime::genCoMin(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value A, mlir::Value resultImage,
                            mlir::Value stat, mlir::Value errmsg) {
  mlir::Type argTy =
      fir::unwrapSequenceType(fir::unwrapPassByRefType(A.getType()));
  if (mlir::isa<fir::CharacterType>(argTy))
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_min_character"));
  else
    genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                            PRIFNAME_SUB("co_min"));
}

/// Generate call to runtime subroutine prif_co_sum
void fir::runtime::genCoSum(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Value A, mlir::Value resultImage,
                            mlir::Value stat, mlir::Value errmsg) {
  genCollectiveSubroutine(builder, loc, A, resultImage, stat, errmsg,
                          PRIFNAME_SUB("co_sum"));
}
