//===-- Assign.cpp -- generate assignment runtime API calls ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/assign.h"

using namespace Fortran::runtime;

void fir::runtime::genAssign(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value destBox, mlir::Value sourceBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Assign)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genAssignPolymorphic(fir::FirOpBuilder &builder,
                                        mlir::Location loc, mlir::Value destBox,
                                        mlir::Value sourceBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(AssignPolymorphic)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genAssignExplicitLengthCharacter(fir::FirOpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Value destBox,
                                                    mlir::Value sourceBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(AssignExplicitLengthCharacter)>(
          loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genAssignTemporary(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value destBox,
                                      mlir::Value sourceBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(AssignTemporary)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genCopyInAssign(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value destBox,
                                   mlir::Value sourceBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(CopyInAssign)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genCopyOutAssign(fir::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Value destBox,
                                    mlir::Value sourceBox) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(CopyOutAssign)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

mlir::Value fir::runtime::genCopyOutReadOnlyMode(fir::FirOpBuilder &builder,
                                                 mlir::Location loc) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(CopyOutReadOnlyMode)>(loc, builder);
  return fir::CallOp::create(builder, loc, func).getResult(0);
}

mlir::Value
fir::runtime::genCopyOutReadOnlyCandidate(fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(CopyOutReadOnlyCandidate)>(
      loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

mlir::Value fir::runtime::genCopyOutReadOnlyConfirm(fir::FirOpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(CopyOutReadOnlyConfirm)>(
      loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  return fir::CallOp::create(builder, loc, func, args).getResult(0);
}

void fir::runtime::genNoteSkippedCopyOut(fir::FirOpBuilder &builder,
                                         mlir::Location loc) {
  auto func =
      fir::runtime::getRuntimeFunc<mkRTKey(NoteSkippedCopyOut)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(1));
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::genAssignSimple(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Value destBox,
                                   mlir::Value sourceBox) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(AssignSimple)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto args = fir::runtime::createArguments(builder, loc, fTy, destBox,
                                            sourceBox, sourceFile, sourceLine);
  fir::CallOp::create(builder, loc, func, args);
}
