//===-- gpuintrin.h - Generic GPU intrinsic functions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPUINTRIN_H
#define __GPUINTRIN_H

#if defined(__NVPTX__)
#include <nvptxintrin.h>
#elif defined(__AMDGPU__)
#include <amdgpuintrin.h>
#endif

#endif // __GPUINTRIN_H
