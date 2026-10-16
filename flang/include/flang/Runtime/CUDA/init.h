//===-- include/flang/Runtime/CUDA/init.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_INIT_H_
#define FORTRAN_RUNTIME_CUDA_INIT_H_

#include "flang/Runtime/entry-names.h"

extern "C" {

void RTDECL(CUFInit)();
}

#endif // FORTRAN_RUNTIME_CUDA_INIT_H_
