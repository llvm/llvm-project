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

#include "flang/Optimizer/OpenACC/Support/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCTypeInterfaces.h"

namespace fir::acc {
void registerOpenACCExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            fir::FIROpsDialect *dialect) {
    fir::BoxType::attachInterface<OpenACCMappableModel<fir::BaseBoxType>>(*ctx);
    fir::ClassType::attachInterface<OpenACCMappableModel<fir::BaseBoxType>>(
        *ctx);
    fir::ReferenceType::attachInterface<
        OpenACCMappableModel<fir::ReferenceType>>(*ctx);
    fir::PointerType::attachInterface<OpenACCMappableModel<fir::PointerType>>(
        *ctx);
    fir::HeapType::attachInterface<OpenACCMappableModel<fir::HeapType>>(*ctx);

    fir::ReferenceType::attachInterface<
        OpenACCPointerLikeModel<fir::ReferenceType>>(*ctx);
    fir::PointerType::attachInterface<
        OpenACCPointerLikeModel<fir::PointerType>>(*ctx);
    fir::HeapType::attachInterface<OpenACCPointerLikeModel<fir::HeapType>>(
        *ctx);

    fir::LLVMPointerType::attachInterface<
        OpenACCPointerLikeModel<fir::LLVMPointerType>>(*ctx);
  });
  registerAttrsExtensions(registry);
}

} // namespace fir::acc
