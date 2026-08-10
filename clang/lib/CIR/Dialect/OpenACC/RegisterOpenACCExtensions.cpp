//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for OpenACC extensions as applied to CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenACC/RegisterOpenACCExtensions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/OpenACC/CIROpenACCTypeInterfaces.h"

namespace cir::acc {

void registerOpenACCExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    cir::PointerType::attachInterface<
        OpenACCPointerLikeModel<cir::PointerType>>(*ctx);
  });
}

} // namespace cir::acc
