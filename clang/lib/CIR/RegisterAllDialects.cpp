//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/InitAllDialects.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/OpenACC/RegisterOpenACCExtensions.h"
#include "clang/CIR/Dialect/OpenMP/RegisterOpenMPExtensions.h"

namespace cir {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect, cir::CIRDialect, mlir::DLTIDialect,
                  mlir::omp::OpenMPDialect, mlir::acc::OpenACCDialect>();
  // Register extensions to integrate CIR types with OpenACC and OpenMP.
  cir::omp::registerOpenMPExtensions(registry);
  cir::acc::registerOpenACCExtensions(registry);
}

void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace cir
