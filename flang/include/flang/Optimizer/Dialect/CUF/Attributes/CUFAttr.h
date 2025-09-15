//===-- Optimizer/Dialect/CUF/Attributes/CUFAttr.h -- CUF attributes ------===//
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

#ifndef FORTRAN_OPTIMIZER_DIALECT_CUF_CUFATTR_H
#define FORTRAN_OPTIMIZER_DIALECT_CUF_CUFATTR_H

#include "flang/Support/Fortran.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class StringRef;
}

namespace mlir {
class Operation;
}

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFEnumAttr.h.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h.inc"

namespace cuf {

/// Attribute to mark Fortran entities with the CUDA attribute.
static constexpr llvm::StringRef dataAttrName = "data_attr";
static constexpr llvm::StringRef getDataAttrName() { return "cuf.data_attr"; }
static constexpr llvm::StringRef getProcAttrName() { return "cuf.proc_attr"; }

/// Attribute to carry CUDA launch_bounds values.
static constexpr llvm::StringRef getLaunchBoundsAttrName() {
  return "cuf.launch_bounds";
}

/// Attribute to carry CUDA cluster_dims values.
static constexpr llvm::StringRef getClusterDimsAttrName() {
  return "cuf.cluster_dims";
}

inline cuf::DataAttributeAttr
getDataAttribute(mlir::MLIRContext *mlirContext,
                 std::optional<Fortran::common::CUDADataAttr> cudaAttr) {
  if (cudaAttr) {
    cuf::DataAttribute attr;
    switch (*cudaAttr) {
    case Fortran::common::CUDADataAttr::Constant:
      attr = cuf::DataAttribute::Constant;
      break;
    case Fortran::common::CUDADataAttr::Device:
      attr = cuf::DataAttribute::Device;
      break;
    case Fortran::common::CUDADataAttr::Managed:
      attr = cuf::DataAttribute::Managed;
      break;
    case Fortran::common::CUDADataAttr::Pinned:
      attr = cuf::DataAttribute::Pinned;
      break;
    case Fortran::common::CUDADataAttr::Shared:
      attr = cuf::DataAttribute::Shared;
      break;
    case Fortran::common::CUDADataAttr::Texture:
      // Obsolete attribute
      return {};
    case Fortran::common::CUDADataAttr::Unified:
      attr = cuf::DataAttribute::Unified;
      break;
    }
    return cuf::DataAttributeAttr::get(mlirContext, attr);
  }
  return {};
}

inline cuf::ProcAttributeAttr
getProcAttribute(mlir::MLIRContext *mlirContext,
                 std::optional<Fortran::common::CUDASubprogramAttrs> cudaAttr) {
  if (cudaAttr) {
    cuf::ProcAttribute attr;
    switch (*cudaAttr) {
    case Fortran::common::CUDASubprogramAttrs::Host:
      attr = cuf::ProcAttribute::Host;
      break;
    case Fortran::common::CUDASubprogramAttrs::Device:
      attr = cuf::ProcAttribute::Device;
      break;
    case Fortran::common::CUDASubprogramAttrs::HostDevice:
      attr = cuf::ProcAttribute::HostDevice;
      break;
    case Fortran::common::CUDASubprogramAttrs::Global:
      attr = cuf::ProcAttribute::Global;
      break;
    case Fortran::common::CUDASubprogramAttrs::Grid_Global:
      attr = cuf::ProcAttribute::GridGlobal;
      break;
    }
    return cuf::ProcAttributeAttr::get(mlirContext, attr);
  }
  return {};
}

/// Returns the data attribute if the operation has one.
cuf::DataAttributeAttr getDataAttr(mlir::Operation *op);

/// Returns true if the operation has a data attribute with the given value.
bool hasDataAttr(mlir::Operation *op, cuf::DataAttribute value);

} // namespace cuf

#endif // FORTRAN_OPTIMIZER_DIALECT_CUF_CUFATTR_H
