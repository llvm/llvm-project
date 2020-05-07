//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "RTBuilder.h"
#include "flang/Lower/FIRBuilder.h"
#include "llvm/ADT/SmallVector.h"

#include "../runtime/stop.h"

using Fortran::lower::operator""_rt_ident;

#define MakeRuntimeEntry(X) mkKey(RTNAME(X))

template <typename RuntimeEntry>
static mlir::FuncOp genRuntimeFunction(Fortran::lower::FirOpBuilder &builder) {
  auto func = builder.getNamedFunction(RuntimeEntry::name);
  if (func)
    return func;
  auto funTy = RuntimeEntry::getTypeModel()(builder.getContext());
  func = builder.createFunction(RuntimeEntry::name, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  return func;
}

mlir::FuncOp
Fortran::lower::genStopStatementRuntime(Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(StopStatement)>(builder);
}

mlir::FuncOp Fortran::lower::genStopStatementTextRuntime(
    Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(StopStatementText)>(builder);
}

mlir::FuncOp Fortran::lower::genFailImageStatementRuntime(
    Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(FailImageStatement)>(builder);
}

mlir::FuncOp Fortran::lower::genProgramEndStatementRuntime(
    Fortran::lower::FirOpBuilder &builder) {
  return genRuntimeFunction<MakeRuntimeEntry(ProgramEndStatement)>(builder);
}
