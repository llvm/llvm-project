//===- ShapedOpInterfaces.cpp - Interfaces for Shaped Ops -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Interfaces/ShapedOpInterfaces.h"

using namespace aiir;

//===----------------------------------------------------------------------===//
// ShapedDimOpInterface
//===----------------------------------------------------------------------===//

LogicalResult aiir::detail::verifyShapedDimOpInterface(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitError("expected single op result");
  if (!op->getResult(0).getType().isIndex())
    return op->emitError("expect index result type");
  return success();
}

/// Include the definitions of the interface.
#include "aiir/Interfaces/ShapedOpInterfaces.cpp.inc"
