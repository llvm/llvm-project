//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for OpenMP extensions as applied to CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenMP/RegisterOpenMPExtensions.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace {
struct OpenMPPointerLikeModel
    : public mlir::omp::PointerLikeType::ExternalModel<OpenMPPointerLikeModel,
                                                       cir::PointerType> {
  mlir::Type getElementType(mlir::Type pointer) const {
    return mlir::cast<cir::PointerType>(pointer).getPointee();
  }
};
} // namespace

namespace cir::omp {

void registerOpenMPExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    cir::FuncOp::attachInterface<
        mlir::omp::DeclareTargetDefaultModel<cir::FuncOp>>(*ctx);
    cir::PointerType::attachInterface<OpenMPPointerLikeModel>(*ctx);
  });
}

} // namespace cir::omp
