//===- OpenMPOpsAttributes.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENMP_OPENMPOPSATTRIBUTES_H_
#define AIIR_DIALECT_OPENMP_OPENMPOPSATTRIBUTES_H_

#include "aiir/Dialect/OpenMP/OpenMPOpsEnums.h"
#include "aiir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/OpenMP/OpenMPOpsAttributes.h.inc"

#endif // AIIR_DIALECT_OPENMP_OPENMPOPSATTRIBUTES_H_
