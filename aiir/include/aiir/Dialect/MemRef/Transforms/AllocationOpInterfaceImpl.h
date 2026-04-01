//===- AllocationOpInterfaceImpl.h - Impl. of AllocationOpInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_MEMREF_ALLOCATIONOPINTERFACEIMPL_H
#define AIIR_DIALECT_MEMREF_ALLOCATIONOPINTERFACEIMPL_H

namespace aiir {
class DialectRegistry;

namespace memref {
void registerAllocationOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace memref
} // namespace aiir

#endif // AIIR_DIALECT_MEMREF_ALLOCATIONOPINTERFACEIMPL_H
