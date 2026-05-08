//===-- include/flang/Runtime/CUDA/registration.h ---------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_REGISTRATION_H_
#define FORTRAN_RUNTIME_CUDA_REGISTRATION_H_

#include "flang/Runtime/entry-names.h"
#include <cstddef>
#include <cstdint>

namespace Fortran::runtime::cuda {

extern "C" {

/// Register a CUDA module.
void *RTDECL(CUFRegisterModule)(void *data);

/// Register a device function.
void RTDECL(CUFRegisterFunction)(
    void **module, const char *fctSym, char *fctName);

/// Register a device variable.
void RTDECL(CUFRegisterVariable)(
    void **module, char *varSym, const char *varName, int64_t size);

/// Register a module-scope variable as host-resident under -gpu=mem:unified,
/// so that the device-side symbol of the same name is mapped to the host
/// pointer at module-load time. Wraps __cudaRegisterHostVar. Kernel accesses
/// to the variable then reach the host storage directly via HMM/ATS.
void RTDECL(CUFRegisterExternalVariable)(
    void **module, char *varSym, const char *varName, int64_t size);

/// Register a managed variable.
void RTDECL(CUFRegisterManagedVariable)(
    void **module, void **varSym, char *varName, int64_t size);

/// Initialize a CUDA module after all variables have been registered.
/// Triggers the runtime to populate managed variable pointers with
/// unified memory addresses.
void RTDECL(CUFInitModule)(void **module);

} // extern "C"

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_REGISTRATION_H_
