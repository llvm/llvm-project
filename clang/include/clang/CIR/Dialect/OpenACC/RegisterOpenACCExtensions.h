//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_OPENACC_REGISTEROPENACCEXTENSIONS_H
#define CLANG_CIR_DIALECT_OPENACC_REGISTEROPENACCEXTENSIONS_H

namespace aiir {
class DialectRegistry;
} // namespace aiir

namespace cir::acc {

void registerOpenACCExtensions(aiir::DialectRegistry &registry);

} // namespace cir::acc

#endif // CLANG_CIR_DIALECT_OPENACC_REGISTEROPENACCEXTENSIONS_H
