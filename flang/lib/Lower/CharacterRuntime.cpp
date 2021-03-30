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
  auto lptr = builder.createConvert(loc, fTy.getInput(0), lhsBuff);
  auto llen = builder.createConvert(loc, fTy.getInput(2), lhsLen);
  auto rptr = builder.createConvert(loc, fTy.getInput(1), rhsBuff);
  auto rlen = builder.createConvert(loc, fTy.getInput(3), rhsLen);
  llvm::SmallVector<mlir::Value> args = {lptr, rptr, llen, rlen};
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
  llvm::SmallVector<mlir::Value> args = {
      builder.createConvert(loc, fTy.getInput(0), stringBase),
      builder.createConvert(loc, fTy.getInput(1), stringLen),
      builder.createConvert(loc, fTy.getInput(2), substringBase),
      builder.createConvert(loc, fTy.getInput(3), substringLen),
      builder.createConvert(loc, fTy.getInput(4), back)};
  return builder.create<fir::CallOp>(loc, indexFunc, args).getResult(0);
}

void Fortran::lower::genIndexDescriptor(Fortran::lower::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value resultBox,
                                        mlir::Value stringBox,
                                        mlir::Value substringBox,
                                        mlir::Value backOpt, mlir::Value kind) {
  auto indexFunc = getRuntimeFunc<mkRTKey(Index)>(loc, builder);
  auto fTy = indexFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(6));

  llvm::SmallVector<mlir::Value> args;
  args.emplace_back(builder.createConvert(loc, fTy.getInput(0), resultBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(1), stringBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(2), substringBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(3), backOpt));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(4), kind));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(5), sourceFile));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(6), sourceLine));
  builder.create<fir::CallOp>(loc, indexFunc, args);
}

void Fortran::lower::genTrim(Fortran::lower::FirOpBuilder &builder,
                             mlir::Location loc, mlir::Value resultBox,
                             mlir::Value stringBox) {
  auto trimFunc = getRuntimeFunc<mkRTKey(Trim)>(loc, builder);
  auto fTy = trimFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(3));

  llvm::SmallVector<mlir::Value> args;
  args.emplace_back(builder.createConvert(loc, fTy.getInput(0), resultBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(1), stringBox));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(2), sourceFile));
  args.emplace_back(builder.createConvert(loc, fTy.getInput(3), sourceLine));
  builder.create<fir::CallOp>(loc, trimFunc, args);
}

/// generate call to Scan runtime library routine. 
/// TODO: Add specialized kind versions.
void
Fortran::lower::genScan(Fortran::lower::FirOpBuilder &builder,
                             mlir::Location loc, mlir::Value resultBox,
                             mlir::Value stringBox, mlir::Value setBox,
                             mlir::Value backBox, mlir::Value kind) {
  auto scanFunc = getRuntimeFunc<mkRTKey(Scan)>(loc, builder);
  auto fTy = scanFunc.getType();
  auto sourceFile = Fortran::lower::locationToFilename(builder, loc);
  auto sourceLine =
      Fortran::lower::locationToLineNo(builder, loc, fTy.getInput(6));

  llvm::SmallVector<mlir::Value> args = {
    builder.createConvert(loc, fTy.getInput(0), resultBox),
    builder.createConvert(loc, fTy.getInput(1), stringBox),
    builder.createConvert(loc, fTy.getInput(2), setBox),
    builder.createConvert(loc, fTy.getInput(3), backBox),
    builder.createConvert(loc, fTy.getInput(4), kind),
    builder.createConvert(loc, fTy.getInput(5), sourceFile),
    builder.createConvert(loc, fTy.getInput(6), sourceLine) 
  };
   
  builder.create<fir::CallOp>(loc, scanFunc, args);
}
