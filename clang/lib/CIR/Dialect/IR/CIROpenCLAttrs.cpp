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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// OpenCLKernelMetadataAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OpenCLKernelMetadataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayAttr workGroupSizeHint, ArrayAttr reqdWorkGroupSize,
    TypeAttr vecTypeHint, std::optional<bool> vecTypeHintSignedness,
    IntegerAttr intelReqdSubGroupSize) {
  // If no field is present, the attribute is considered invalid.
  if (!workGroupSizeHint && !reqdWorkGroupSize && !vecTypeHint &&
      !vecTypeHintSignedness && !intelReqdSubGroupSize) {
    return emitError()
           << "metadata attribute without any field present is invalid";
  }

  // Check for 3-dim integer tuples
  auto is3dimIntTuple = [](ArrayAttr arr) {
    auto isInt = [](Attribute dim) { return mlir::isa<IntegerAttr>(dim); };
    return arr.size() == 3 && llvm::all_of(arr, isInt);
  };
  if (workGroupSizeHint && !is3dimIntTuple(workGroupSizeHint)) {
    return emitError()
           << "work_group_size_hint must have exactly 3 integer elements";
  }
  if (reqdWorkGroupSize && !is3dimIntTuple(reqdWorkGroupSize)) {
    return emitError()
           << "reqd_work_group_size must have exactly 3 integer elements";
  }

  // Check for co-presence of vecTypeHintSignedness
  if (!!vecTypeHint != vecTypeHintSignedness.has_value()) {
    return emitError() << "vec_type_hint_signedness should be present if and "
                          "only if vec_type_hint is set";
  }

  if (vecTypeHint) {
    Type vecTypeHintValue = vecTypeHint.getValue();
    if (mlir::isa<cir::CIRDialect>(vecTypeHintValue.getDialect())) {
      // Check for signedness alignment in CIR
      if (isSignedHint(vecTypeHintValue) != vecTypeHintSignedness) {
        return emitError() << "vec_type_hint_signedness must match the "
                              "signedness of the vec_type_hint type";
      }
      // Check for the dialect of type hint
    } else if (!LLVM::isCompatibleType(vecTypeHintValue)) {
      return emitError() << "vec_type_hint must be a type from the CIR or LLVM "
                            "dialect";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OpenCLKernelArgMetadataAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OpenCLKernelArgMetadataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayAttr addrSpaces, ArrayAttr accessQuals, ArrayAttr types,
    ArrayAttr baseTypes, ArrayAttr typeQuals, ArrayAttr argNames) {
  auto isIntArray = [](ArrayAttr elt) {
    return llvm::all_of(
        elt, [](Attribute elt) { return mlir::isa<IntegerAttr>(elt); });
  };
  auto isStrArray = [](ArrayAttr elt) {
    return llvm::all_of(
        elt, [](Attribute elt) { return mlir::isa<StringAttr>(elt); });
  };

  if (!isIntArray(addrSpaces))
    return emitError() << "addr_space must be integer arrays";
  if (!llvm::all_of<ArrayRef<ArrayAttr>>(
          {accessQuals, types, baseTypes, typeQuals}, isStrArray))
    return emitError()
           << "access_qual, type, base_type, type_qual must be string arrays";
  if (argNames && !isStrArray(argNames)) {
    return emitError() << "name must be a string array";
  }

  if (!llvm::all_of<ArrayRef<ArrayAttr>>(
          {addrSpaces, accessQuals, types, baseTypes, typeQuals, argNames},
          [&](ArrayAttr arr) {
            return !arr || arr.size() == addrSpaces.size();
          })) {
    return emitError() << "all arrays must have the same number of elements";
  }
  return success();
}
