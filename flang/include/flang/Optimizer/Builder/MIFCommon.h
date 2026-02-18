//===-- MIFCommon.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MIFCOMMON_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_MIFCOMMON_H_

#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Runtime/coarray.h"
#include "mlir/IR/BuiltinOps.h"

static constexpr llvm::StringRef coarrayHandleSuffix = "_coarray_handle";

namespace mif {

std::string getFullUniqName(mlir::Value addr);

} // namespace mif

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MIFCOMMON_H_
