//===-- CharacterRuntime.cpp -- runtime for CHARACTER type entities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CharacterRuntime.h"
#include "../../runtime/character.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

/// Generate calls to string handling intrinsics such as index, scan, and
/// verify. These are the descriptor based implementations that take four
/// arguments (string1, string2, back, kind).
template <typename FN>
static void genCharacterSearch(FN func, Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value string1Box, mlir::Value string2Box,
                               mlir::Value backBox, mlir::Value kind);

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

mlir::Value
Fortran::lower::genRawCharCompare(Fortran::lower::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::CmpIPredicate cmp,
                                  mlir::Value lhsBuff, mlir::Value lhsLen,
                                  mlir::Value rhsBuff, mlir::Value rhsLen) {
  mlir::FuncOp beginFunc;
  switch (discoverKind(lhsBuff.getType())) {
  case 1:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar1)>(loc, builder);
    break;
  case 2:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar2)>(loc, builder);
    break;
  case 4:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar4)>(loc, builder);
    break;
  default:
    llvm_unreachable("runtime does not support CHARACTER KIND");
  }
  auto fTy = beginFunc.getType();
  auto args = Fortran::lower::createArguments(builder, loc, fTy, lhsBuff,
                                              lhsLen, rhsBuff, rhsLen);
  auto tri = builder.create<fir::CallOp>(loc, beginFunc, args).getResult(0);
  auto zero = builder.createIntegerConstant(loc, tri.getType(), 0);
  return builder.create<mlir::CmpIOp>(loc, cmp, tri, zero);
}

mlir::Value
Fortran::lower::genCharCompare(Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::CmpIPredicate cmp,
                               const fir::ExtendedValue &lhs,
                               const fir::ExtendedValue &rhs) {
  if (lhs.getBoxOf<fir::BoxValue>() || rhs.getBoxOf<fir::BoxValue>())
    TODO(loc, "character compare from descriptors");
  auto allocateIfNotInMemory = [&](mlir::Value base) -> mlir::Value {
    if (fir::isa_ref_type(base.getType()))
      return base;
    auto mem = builder.create<fir::AllocaOp>(loc, base.getType());
    builder.create<fir::StoreOp>(loc, base, mem);
    return mem;
  };
  auto lhsBuffer = allocateIfNotInMemory(fir::getBase(lhs));
  auto rhsBuffer = allocateIfNotInMemory(fir::getBase(rhs));
  return genRawCharCompare(builder, loc, cmp, lhsBuffer, fir::getLen(lhs),
                           rhsBuffer, fir::getLen(rhs));
}

mlir::Value
Fortran::lower::genIndex(Fortran::lower::FirOpBuilder &builder,
                         mlir::Location loc, int kind, mlir::Value stringBase,
                         mlir::Value stringLen, mlir::Value substringBase,
                         mlir::Value substringLen, mlir::Value back) {
  mlir::FuncOp indexFunc;
  switch (kind) {
  case 1:
    indexFunc = getRuntimeFunc<mkRTKey(Index1)>(loc, builder);
    break;
  case 2:
    indexFunc = getRuntimeFunc<mkRTKey(Index2)>(loc, builder);
    break;
  case 4:
    indexFunc = getRuntimeFunc<mkRTKey(Index4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
  }
  auto fTy = indexFunc.getType();
  auto args =
      Fortran::lower::createArguments(builder, loc, fTy, stringBase, stringLen,
                                      substringBase, substringLen, back);
  return builder.create<fir::CallOp>(loc, indexFunc, args).getResult(0);
}

void Fortran::lower::genIndexDescriptor(Fortran::lower::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value resultBox,
                                        mlir::Value stringBox,
                                        mlir::Value substringBox,
                                        mlir::Value backOpt, mlir::Value kind) {
  auto indexFunc = getRuntimeFunc<mkRTKey(Index)>(loc, builder);
  genCharacterSearch(indexFunc, builder, loc, resultBox, stringBox,
                     substringBox, backOpt, kind);
}

void Fortran::lower::genTrim(Fortran::lower::FirOpBuilder &builder,
                             mlir::Location loc, mlir::Value resultBox,
                             mlir::Value stringBox) {
  auto trimFunc = getRuntimeFunc<mkRTKey(Trim)>(loc, builder);
  auto fTy = trimFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(3));

  auto args = Fortran::lower::createArguments(
      builder, loc, fTy, resultBox, stringBox, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, trimFunc, args);
}

/// Generate call to scan runtime routine.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsic.
void Fortran::lower::genScanDescriptor(Fortran::lower::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value resultBox,
                                       mlir::Value stringBox,
                                       mlir::Value setBox, mlir::Value backBox,
                                       mlir::Value kind) {
  auto func = getRuntimeFunc<mkRTKey(Scan)>(loc, builder);
  genCharacterSearch(func, builder, loc, resultBox, stringBox, setBox, backBox,
                     kind);
}

/// Generate call to scan runtime routine that is specialized on
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
mlir::Value Fortran::lower::genScan(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, int kind,
                                    mlir::Value stringBase,
                                    mlir::Value stringLen, mlir::Value setBase,
                                    mlir::Value setLen, mlir::Value back) {
  mlir::FuncOp func;
  switch (kind) {
  case 1:
    func = getRuntimeFunc<mkRTKey(Scan1)>(loc, builder);
    break;
  case 2:
    func = getRuntimeFunc<mkRTKey(Scan2)>(loc, builder);
    break;
  case 4:
    func = getRuntimeFunc<mkRTKey(Scan4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
  }
  auto fTy = func.getType();
  auto args = Fortran::lower::createArguments(builder, loc, fTy, stringBase,
                                              stringLen, setBase, setLen, back);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate call to verify runtime routine.
/// This calls the descriptor based runtime call implementation of the
/// verify intrinsic.
void Fortran::lower::genVerifyDescriptor(
    Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
    mlir::Value resultBox, mlir::Value stringBox, mlir::Value setBox,
    mlir::Value backBox, mlir::Value kind) {
  auto func = getRuntimeFunc<mkRTKey(Verify)>(loc, builder);
  genCharacterSearch(func, builder, loc, resultBox, stringBox, setBox, backBox,
                     kind);
}

/// Generate call to verify runtime routine that is specialized on
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
mlir::Value Fortran::lower::genVerify(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, int kind,
                                      mlir::Value stringBase,
                                      mlir::Value stringLen,
                                      mlir::Value setBase, mlir::Value setLen,
                                      mlir::Value back) {
  mlir::FuncOp func;
  switch (kind) {
  case 1:
    func = getRuntimeFunc<mkRTKey(Verify1)>(loc, builder);
    break;
  case 2:
    func = getRuntimeFunc<mkRTKey(Verify2)>(loc, builder);
    break;
  case 4:
    func = getRuntimeFunc<mkRTKey(Verify4)>(loc, builder);
    break;
  default:
    fir::emitFatalError(
        loc, "unsupported CHARACTER kind value. Runtime expects 1, 2, or 4.");
  }
  auto fTy = func.getType();
  auto args = Fortran::lower::createArguments(builder, loc, fTy, stringBase,
                                              stringLen, setBase, setLen, back);
  return builder.create<fir::CallOp>(loc, func, args).getResult(0);
}

/// Generate calls to string handling intrinsics such as index, scan, and
/// verify. These are the descriptor based implementations that take four
/// arguments (string1, string2, back, kind).
template <typename FN>
static void genCharacterSearch(FN func, Fortran::lower::FirOpBuilder &builder,
                               mlir::Location loc, mlir::Value resultBox,
                               mlir::Value string1Box, mlir::Value string2Box,
                               mlir::Value backBox, mlir::Value kind) {

  auto fTy = func.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(6));

  auto args = Fortran::lower::createArguments(builder, loc, fTy, resultBox,
                                              string1Box, string2Box, backBox,
                                              kind, sourceFile, sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}
