//===- RegisterOpenMPExtensions.h - OpenMP Extension Registration -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENMP_SUPPORT_REGISTEROPENMPEXTENSIONS_H_
#define FLANG_OPTIMIZER_OPENMP_SUPPORT_REGISTEROPENMPEXTENSIONS_H_

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace fir::omp {

void registerOpenMPExtensions(mlir::DialectRegistry &registry);

/// Register external models for FIR attributes related to OpenMP.
void registerAttrsExtensions(mlir::DialectRegistry &registry);

/// Register all dialects whose operations may be created
/// by the transformational attributes.
void registerTransformationalAttrsDependentDialects(
    mlir::DialectRegistry &registry);

/// Register external models for FIR operation interfaces related to OpenMP.
void registerOpInterfacesExtensions(mlir::DialectRegistry &registry);

} // namespace fir::omp

#endif // FLANG_OPTIMIZER_OPENMP_SUPPORT_REGISTEROPENMPEXTENSIONS_H_
