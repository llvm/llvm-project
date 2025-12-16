//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_OPENCL_BASE_H__
#define __CLC_OPENCL_OPENCL_BASE_H__

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

/* Function Attributes */
#include <clc/clcfunc.h>

/* 6.1 Supported Data Types */
#include <clc/clctypes.h>

#endif // __CLC_OPENCL_OPENCL_BASE_H__
