//===-- include/flang/Runtime/CUDA/stream.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_STREAM_H_
#define FORTRAN_RUNTIME_CUDA_STREAM_H_

#include "common.h"
#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

extern "C" {

int RTDECL(CUFSetDefaultStream)(cudaStream_t);
cudaStream_t RTDECL(CUFGetDefaultStream)();
int RTDECL(CUFStreamSynchronize)(cudaStream_t);
int RTDECL(CUFStreamSynchronizeNull)();
}

} // namespace Fortran::runtime::cuda
#endif // FORTRAN_RUNTIME_CUDA_STREAM_H_
