//===- CIROpenCLAttrs.cpp - OpenCL specific attributes in CIR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpenCL-specific attrs in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// OpenCLKernelArgMetadataAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OpenCLKernelArgMetadataAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr addrSpaces,
    ArrayAttr accessQuals, ArrayAttr types, ArrayAttr baseTypes,
    ArrayAttr typeQuals, ArrayAttr argNames) {
  if (!llvm::all_of(ArrayRef<ArrayAttr>{addrSpaces, accessQuals, types,
                                        baseTypes, typeQuals, argNames},
                    [&](ArrayAttr attr) {
                      return !attr || attr.size() == addrSpaces.size();
                    }))
    return emitError() << "all arrays must have the same number of elements";

  return success();
}
