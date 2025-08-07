//===-- RegisterOpenACCExtensions.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for OpenACC extensions as applied to FIR dialect.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/FIROpenACCTypeInterfaces.h"

namespace fir::acc {
void registerOpenACCExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            fir::FIROpsDialect *dialect) {
    fir::SequenceType::attachInterface<OpenACCMappableModel<fir::SequenceType>>(
        *ctx);
    fir::BoxType::attachInterface<OpenACCMappableModel<fir::BaseBoxType>>(*ctx);
  });
}

} // namespace fir::acc
