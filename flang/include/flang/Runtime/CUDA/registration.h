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

namespace Fortran::runtime::cuda {

extern "C" {

/// Register a CUDA module.
void *RTDECL(CUFRegisterModule)(void *data);

/// Register a device function.
void RTDECL(CUFRegisterFunction)(void **module, const char *fct);

} // extern "C"

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_REGISTRATION_H_
