//===-- Derived.cpp -- derived type runtime API ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/derived-api.h"
#include "flang/Runtime/pointer.h"

using namespace Fortran::runtime;

void fir::runtime::genDerivedTypeInitialize(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Initialize)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto sourceFile = fir::factory::locationToFilename(builder, loc);
  auto sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  auto args = fir::runtime::createArguments(builder, loc, fTy, box, sourceFile,
                                            sourceLine);
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::genDerivedTypeDestroy(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value box) {
  auto func = fir::runtime::getRuntimeFunc<mkRTKey(Destroy)>(loc, builder);
  auto fTy = func.getFunctionType();
  auto args = fir::runtime::createArguments(builder, loc, fTy, box);
  builder.create<fir::CallOp>(loc, func, args);
}

void fir::runtime::genNullifyDerivedType(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value box,
                                         fir::RecordType derivedType,
                                         unsigned rank) {
  std::string typeDescName =
      fir::NameUniquer::getTypeDescriptorName(derivedType.getName());
  fir::GlobalOp typeDescGlobal = builder.getNamedGlobal(typeDescName);
  if (!typeDescGlobal)
    fir::emitFatalError(loc, "no type descriptor found for NULLIFY");
  auto typeDescAddr = builder.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(typeDescGlobal.getType()),
      typeDescGlobal.getSymbol());
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(PointerNullifyDerived)>(loc,
                                                                   builder);
  llvm::ArrayRef<mlir::Type> inputTypes = callee.getFunctionType().getInputs();
  llvm::SmallVector<mlir::Value> args;
  args.push_back(builder.createConvert(loc, inputTypes[0], box));
  args.push_back(builder.createConvert(loc, inputTypes[1], typeDescAddr));
  mlir::Value rankCst = builder.createIntegerConstant(loc, inputTypes[2], rank);
  mlir::Value c0 = builder.createIntegerConstant(loc, inputTypes[3], 0);
  args.push_back(rankCst);
  args.push_back(c0);
  builder.create<fir::CallOp>(loc, callee, args);
}
