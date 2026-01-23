//===- SafeTempArrayCopyAttrInterface.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_SAFETEMPARRAYCOPYATTRINTERFACE_H
#define FORTRAN_OPTIMIZER_DIALECT_SAFETEMPARRAYCOPYATTRINTERFACE_H

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace fir {
class FirOpBuilder;
}

#include "flang/Optimizer/Dialect/SafeTempArrayCopyAttrInterface.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_SAFETEMPARRAYCOPYATTRINTERFACE_H
