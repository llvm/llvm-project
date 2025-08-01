//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_SYNCHRONIZATION_BARRIER_H__
#define __CLC_OPENCL_SYNCHRONIZATION_BARRIER_H__

#include <clc/opencl/opencl-base.h>
#include <clc/opencl/synchronization/cl_mem_fence_flags.h>

_CLC_DECL _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags);

#endif // __CLC_OPENCL_SYNCHRONIZATION_BARRIER_H__
