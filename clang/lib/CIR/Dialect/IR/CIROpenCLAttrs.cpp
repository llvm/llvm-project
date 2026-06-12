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
#include "clang/CIR/Dialect/IR/CIRTypes.h"

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

//===----------------------------------------------------------------------===//
// OpenCLKernelExecutionMetadataAttr definitions
//===----------------------------------------------------------------------===//

static LogicalResult
verifyWorkGroupSizeMetadata(function_ref<InFlightDiagnostic()> emitError,
                            DenseI32ArrayAttr metadata, StringRef name) {
  if (!metadata)
    return success();
  if (metadata.size() != 3)
    return emitError() << name << " must have exactly three elements";

  ArrayRef<int32_t> values = metadata.asArrayRef();
  bool allZero = llvm::all_of(values, [](int32_t value) { return value == 0; });
  bool allNonZero =
      llvm::all_of(values, [](int32_t value) { return value != 0; });
  if (!allZero && !allNonZero)
    return emitError() << name << " elements must be all zero or all non-zero";

  return success();
}

LogicalResult OpenCLKernelExecutionMetadataAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    DenseI32ArrayAttr reqdWorkGroupSize, DenseI32ArrayAttr workGroupSizeHint,
    TypeAttr vecTypeHint) {
  if (!reqdWorkGroupSize && !workGroupSizeHint && !vecTypeHint)
    return emitError()
           << "kernel execution metadata requires at least one field";
  if (failed(verifyWorkGroupSizeMetadata(emitError, reqdWorkGroupSize,
                                         "reqd_work_group_size")))
    return failure();
  if (failed(verifyWorkGroupSizeMetadata(emitError, workGroupSizeHint,
                                         "work_group_size_hint")))
    return failure();

  if (vecTypeHint) {
    Type type = vecTypeHint.getValue();
    if (auto vectorType = dyn_cast<cir::VectorType>(type);
        vectorType && !vectorType.isScalable())
      type = vectorType.getElementType();
    if (!isa<cir::IntType, cir::FPTypeInterface>(type))
      return emitError()
             << "vec_type_hint must be a CIR integer or floating-point "
                "type, or a fixed vector of such a type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OpenCLVersionAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
OpenCLVersionAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          int32_t majorVersion, int32_t minorVersion) {
  if (majorVersion <= 0)
    return emitError() << "OpenCL major version must be positive";
  if (minorVersion < 0)
    return emitError() << "OpenCL minor version must be non-negative";

  return success();
}
