//===- ValueBoundsOpInterfaceImpl.h - Impl. of ValueBoundsOpInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_IR_VALUEBOUNDSOPINTERFACEIMPL_H
#define AIIR_DIALECT_MEMREF_IR_VALUEBOUNDSOPINTERFACEIMPL_H

namespace aiir {
class DialectRegistry;

namespace memref {
void registerValueBoundsOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace memref
} // namespace aiir

#endif // AIIR_DIALECT_MEMREF_IR_VALUEBOUNDSOPINTERFACEIMPL_H
