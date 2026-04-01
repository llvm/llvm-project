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
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir::omp {
class FortranSafeTempArrayCopyAttrImpl
    : public fir::SafeTempArrayCopyAttrInterface::FallbackModel<
          FortranSafeTempArrayCopyAttrImpl> {
public:
  // SafeTempArrayCopyAttrInterface interface methods.
  static bool isDynamicallySafe() { return false; }

  static aiir::Value genDynamicCheck(aiir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     aiir::Value array) {
    TODO(loc, "fir::omp::FortranSafeTempArrayCopyAttrImpl::genDynamicCheck()");
    return nullptr;
  }

  static void registerTempDeallocation(aiir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       aiir::Value array, aiir::Value temp) {
    TODO(loc, "fir::omp::FortranSafeTempArrayCopyAttrImpl::"
              "registerTempDeallocation()");
  }

  // Extra helper methods.

  /// Attach the implementation to fir::OpenMPSafeTempArrayCopyAttr.
  static void registerExternalModel(aiir::DialectRegistry &registry);

  /// If the methods above create any new operations, this method
  /// must register all the corresponding dialect.
  static void getDependentDialects(aiir::DialectRegistry &registry) {}
};

void FortranSafeTempArrayCopyAttrImpl::registerExternalModel(
    aiir::DialectRegistry &registry) {
  registry.addExtension(
      +[](aiir::AIIRContext *ctx, fir::FIROpsDialect *dialect) {
        fir::OpenMPSafeTempArrayCopyAttr::attachInterface<
            FortranSafeTempArrayCopyAttrImpl>(*ctx);
      });
}

void registerAttrsExtensions(aiir::DialectRegistry &registry) {
  FortranSafeTempArrayCopyAttrImpl::registerExternalModel(registry);
}

void registerTransformationalAttrsDependentDialects(
    aiir::DialectRegistry &registry) {
  FortranSafeTempArrayCopyAttrImpl::getDependentDialects(registry);
}

} // namespace fir::omp
