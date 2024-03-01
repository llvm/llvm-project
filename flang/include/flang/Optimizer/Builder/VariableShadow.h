//===-- VariableShadow.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "mlir/IR/Value.h"

#ifndef FORTRAN_OPTIMIZER_BUILDER_VARIABLESHADOW_H
#define FORTRAN_OPTIMIZER_BUILDER_VARIABLESHADOW_H

namespace hlfir {

class FortranVariableShadow {
public:
  FortranVariableShadow(fir::FirOpBuilder &builder,
                        mlir::BlockArgument shadowingVal,
                        fir::FortranVariableOpInterface shadowedVariable);

  mlir::BlockArgument getValue() const { return shadowingVal; }

  mlir::Value getBase() const { return base; }

  mlir::Value getFirBase() const { return firBase; }

  fir::FortranVariableOpInterface getShadowedVariable() const {
    return shadowedVariable;
  }

private:
  mlir::BlockArgument shadowingVal;
  mlir::Value base;
  mlir::Value firBase;
  fir::FortranVariableOpInterface shadowedVariable;
};

} // namespace hlfir

#endif // FORTRAN_OPTIMIZER_BUILDER_VARIABLESHADOW_H
