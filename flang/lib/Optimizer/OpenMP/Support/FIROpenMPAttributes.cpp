//===-- FIROpenMPAttributes.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements attribute interfaces that are promised by FIR
/// dialect attributes related to OpenMP.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir::omp {
class FortranSafeTempArrayCopyAttrImpl
    : public fir::SafeTempArrayCopyAttrInterface::FallbackModel<
          FortranSafeTempArrayCopyAttrImpl> {
public:
  // SafeTempArrayCopyAttrInterface interface methods.
  static bool isDynamicallySafe() { return false; }

  static mlir::Value genDynamicCheck(mlir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     mlir::Value array) {
    TODO(loc, "fir::omp::FortranSafeTempArrayCopyAttrImpl::genDynamicCheck()");
    return nullptr;
  }

  static void registerTempDeallocation(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       mlir::Value array, mlir::Value temp) {
    TODO(loc, "fir::omp::FortranSafeTempArrayCopyAttrImpl::"
              "registerTempDeallocation()");
  }

  // Extra helper methods.

  /// Attach the implementation to fir::OpenMPSafeTempArrayCopyAttr.
  static void registerExternalModel(mlir::DialectRegistry &registry);

  /// If the methods above create any new operations, this method
  /// must register all the corresponding dialect.
  static void getDependentDialects(mlir::DialectRegistry &registry) {}
};

void FortranSafeTempArrayCopyAttrImpl::registerExternalModel(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, fir::FIROpsDialect *dialect) {
        fir::OpenMPSafeTempArrayCopyAttr::attachInterface<
            FortranSafeTempArrayCopyAttrImpl>(*ctx);
      });
}

void registerAttrsExtensions(mlir::DialectRegistry &registry) {
  FortranSafeTempArrayCopyAttrImpl::registerExternalModel(registry);
}

void registerTransformationalAttrsDependentDialects(
    mlir::DialectRegistry &registry) {
  FortranSafeTempArrayCopyAttrImpl::getDependentDialects(registry);
}

} // namespace fir::omp
