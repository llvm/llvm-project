//===-- Character.cpp -- runtime for CHARACTER type entities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/character.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace Fortran::runtime;

/// Generate calls to string handling intrinsics such as index, scan, and
/// verify. These are the descriptor based implementations that take four
/// arguments (string1, string2, back, kind).
template <typename FN>
static void genCharacterSearch(FN func, fir::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value string1Box, mlir::Value string2Box,
                               mlir::Value backBox, mlir::Value kind) {

  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(6));

  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            string1Box, string2Box, backBox,
                                            kind, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

/// Helper function to recover the KIND from the FIR type.
static int discoverKind(mlir::Type ty) {
  if (auto charTy = ty.dyn_cast<fir::CharacterType>())
    return charTy.getFKind();
  if (auto eleTy = fir::dyn_cast_ptrEleTy(ty))
    return discoverKind(eleTy);
  if (auto arrTy = ty.dyn_cast<fir::SequenceType>())
    return discoverKind(arrTy.getEleTy());
  if (auto boxTy = ty.dyn_cast<fir::BoxCharType>())
    return discoverKind(boxTy.getEleTy());
  if (auto boxTy = ty.dyn_cast<fir::BoxType>())
    return discoverKind(boxTy.getEleTy());
  llvm_unreachable("unexpected character type");
}

//===----------------------------------------------------------------------===//
// Lower character operations
//===----------------------------------------------------------------------===//

/// Generate a call to the `ADJUST[L|R]` runtime.
///
/// \p resultBox must be an unallocated allocatable used for the temporary
/// result.  \p StringBox must be a fir.box describing the adjustr string
/// argument.  The \p adjustFunc should be a mlir::func::FuncOp for the
/// appropriate runtime entry function.
static void genAdjust(fir::FirOpBuilder &builder, mlir::Location loc,
                      mlir::Value resultBox, mlir::Value stringBox,
                      mlir::func::FuncOp &adjustFunc) {

  auto fTy = adjustFunc.getFunctionType();
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            stringBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, adjustFunc, args);
}

void fir::runtime::genAdjustL(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value stringBox) {
  auto adjustFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Adjustl)>(loc, builder);
  genAdjust(builder, loc, resultBox, stringBox, adjustFunc);
}

void fir::runtime::genAdjustR(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Value resultBox, mlir::Value stringBox) {
  auto adjustFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(Adjustr)>(loc, builder);
  genAdjust(builder, loc, resultBox, stringBox, adjustFunc);
}

mlir::Value
fir::runtime::genCharCompare(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::arith::CmpIPredicate cmp,
                             mlir::Value lhsBuff, mlir::Value lhsLen,
                             mlir::Value rhsBuff, mlir::Value rhsLen) {
  mlir::func::FuncOp beginFunc;
  switch (discoverKind(lhsBuff.getType())) {
  case 1:
    beginFunc = fir::runtime::getRuntimeFunc<mkRTKey(CharacterCompareScalar1)>(
        loc, builder);
    break;
  case 2:
    beginFunc = fir::runtime::getRuntimeFunc<mkRTKey(CharacterCompareScalar2)>(
        loc, builder);
    break;
  case 4:
    beginFunc = fir::runtime::getRuntimeFunc<mkRTKey(CharacterCompareScalar4)>(
        loc, builder);
    break;
  default:
    llvm_unreachable("runtime does not support CHARACTER KIND");
  }
  auto fTy = beginFunc.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, lhsBuff, rhsBuff,
                                            lhsLen, rhsLen);
  auto tri = builder.create<fir::CallOp>(loc, beginFunc, args).getResult(0);
  auto zero = builder.createIntegerConstant(loc, tri.getType(), 0);
  return builder.create<mlir::arith::CmpIOp>(loc, cmp, tri, zero);
}

mlir::Value fir::runtime::genCharCompare(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::arith::CmpIPredicate cmp,
                                         const fir::ExtendedValue &lhs,
                                         const fir::ExtendedValue &rhs) {
  if (lhs.getBoxOf<fir::BoxValue>() || rhs.getBoxOf<fir::BoxValue>())
    TODO(loc, "character compare from descriptors");
  auto allocateIfNotInMemory = [&](mlir::Value base) -> mlir::Value {
    if (fir::isa_ref_type(base.getType()))
      return base;
    auto mem =
        builder.create<fir::AllocaOp>(loc, base.getType(), /*pinned=*/false);
    builder.create<fir::StoreOp>(loc, base, mem);
    return mem;
  };
  auto lhsBuffer = allocateIfNotInMemory(fir::getBase(lhs));
  auto rhsBuffer = allocateIfNotInMemory(fir::getBase(rhs));
  return genCharCompare(builder, loc, cmp, lhsBuffer, fir::getLen(lhs),
                        rhsBuffer, fir::getLen(rhs));
}

mlir::Value fir::runtime::genIndex(fir::FirOpBuilder &builder,
                                   mlir::Location loc, int kind,
                                   mlir::Value stringBase,
                                   mlir::Value stringLen,
                                   mlir::Value substringBase,
                                   mlir::Value substringLen, mlir::Value back) {
  mlir::func::FuncOp indexFunc;
  switch (kind) {
  case 1:
    indexFunc = fir::runtime::getRuntimeFunc<mkRTKey(Index1)>(loc, builder);
    break;
  case 2:
    indexFunc = fir::runtime::getRuntimeFunc<mkRTKey(Index2)>(loc, builder);
    break;
  case 4:
    indexFunc = fir::runtime::getRuntimeFunc<mkRTKey(Index4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
  }
  auto fTy = indexFunc.getFunctionType();
  auto args =
      fir::runtime::createArguments(builder, loc, fTy, stringBase, stringLen,
                                    substringBase, substringLen, back);
  return builder.create<fir::CallOp>(loc, indexFunc, args).getResult(0);
}

void fir::runtime::genIndexDescriptor(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value resultBox,
                                      mlir::Value stringBox,
                                      mlir::Value substringBox,
                                      mlir::Value backOpt, mlir::Value kind) {
  auto indexFunc = fir::runtime::getRuntimeFunc<mkRTKey(Index)>(loc, builder);
  genCharacterSearch(indexFunc, builder, loc, resultBox, stringBox,
                     substringBox, backOpt, kind);
}

void fir::runtime::genRepeat(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value resultBox, mlir::Value stringBox,
                             mlir::Value ncopies) {
  auto repeatFunc = fir::runtime::getRuntimeFunc<mkRTKey(Repeat)>(loc, builder);
  auto fTy = repeatFunc.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));

  auto args = fir::runtime::createArguments(
      builder, loc, fTy, resultBox, stringBox, ncopies, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, repeatFunc, args);
}

void fir::runtime::genTrim(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value resultBox, mlir::Value stringBox) {
  auto trimFunc = fir::runtime::getRuntimeFunc<mkRTKey(Trim)>(loc, builder);
  auto fTy = trimFunc.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));

  auto args = fir::runtime::createArguments(builder, loc, fTy, resultBox,
                                            stringBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, trimFunc, args);
}

void fir::runtime::genScanDescriptor(fir::FirOpBuilder &builder,
                                     mlir::Location loc, mlir::Value resultBox,
                                     mlir::Value stringBox, mlir::Value setBox,
                                     mlir::Value backBox, mlir::Value kind) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Scan)>(loc, builder);
  genCharacterSearch(func, builder, loc, resultBox, stringBox, setBox, backBox,
                     kind);
}

mlir::Value fir::runtime::genScan(fir::FirOpBuilder &builder,
                                  mlir::Location loc, int kind,
                                  mlir::Value stringBase, mlir::Value stringLen,
                                  mlir::Value setBase, mlir::Value setLen,
                                  mlir::Value back) {
  mlir::func::FuncOp func;
  switch (kind) {
  case 1:
    func = fir::runtime::getRuntimeFunc<mkRTKey(Scan1)>(loc, builder);
    break;
  case 2:
    func = fir::runtime::getRuntimeFunc<mkRTKey(Scan2)>(loc, builder);
    break;
  case 4:
    func = fir::runtime::getRuntimeFunc<mkRTKey(Scan4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
  }
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, stringBase,
                                            stringLen, setBase, setLen, back);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

void fir::runtime::genVerifyDescriptor(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value resultBox,
                                       mlir::Value stringBox,
                                       mlir::Value setBox, mlir::Value backBox,
                                       mlir::Value kind) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Verify)>(loc, builder);
  genCharacterSearch(func, builder, loc, resultBox, stringBox, setBox, backBox,
                     kind);
}

mlir::Value fir::runtime::genVerify(fir::FirOpBuilder &builder,
                                    mlir::Location loc, int kind,
                                    mlir::Value stringBase,
                                    mlir::Value stringLen, mlir::Value setBase,
                                    mlir::Value setLen, mlir::Value back) {
  mlir::func::FuncOp func;
  switch (kind) {
  case 1:
    func = fir::runtime::getRuntimeFunc<mkRTKey(Verify1)>(loc, builder);
    break;
  case 2:
    func = fir::runtime::getRuntimeFunc<mkRTKey(Verify2)>(loc, builder);
    break;
  case 4:
    func = fir::runtime::getRuntimeFunc<mkRTKey(Verify4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
  }
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, stringBase,
                                            stringLen, setBase, setLen, back);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}
