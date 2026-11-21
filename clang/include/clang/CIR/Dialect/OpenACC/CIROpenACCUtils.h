//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains CIR OpenACC utilities.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_OPENACC_CIROPENACCUTILS_H
#define CLANG_CIR_DIALECT_OPENACC_CIROPENACCUTILS_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace cir::acc {

/// Returns true if code on the device can reach storage in `space`. This
/// includes host memory the device accesses directly (`offload_global_host`)
/// and `offload_generic`, which may name device memory.
bool isDeviceAccessibleAddressSpace(cir::LangAddressSpace space);

/// Returns true if storage in `space` physically resides on the device. This
/// is a subset of `isDeviceAccessibleAddressSpace`: host memory and generic
/// pointers are reachable but not resident.
bool isDeviceResidentAddressSpace(cir::LangAddressSpace space);

} // namespace cir::acc

#endif // CLANG_CIR_DIALECT_OPENACC_CIROPENACCUTILS_H
