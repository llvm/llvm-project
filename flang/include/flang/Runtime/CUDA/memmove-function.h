//===-- include/flang/Runtime/CUDA/memmove-function.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#ifndef FORTRAN_RUNTIME_CUDA_MEMMOVE_FUNCTION_H_
#define FORTRAN_RUNTIME_CUDA_MEMMOVE_FUNCTION_H_

namespace Fortran::runtime::cuda {

void *MemmoveHostToDevice(void *dst, const void *src, std::size_t count);

void *MemmoveDeviceToHost(void *dst, const void *src, std::size_t count);

void *MemmoveDeviceToDevice(void *dst, const void *src, std::size_t count);

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_MEMMOVE_FUNCTION_H_
