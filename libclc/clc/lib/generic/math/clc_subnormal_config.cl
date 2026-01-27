//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_subnormal_config.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEF bool __clc_fp16_subnormals_supported() {
#ifdef CLC_SPIRV
  half x = __clc_fabs(0x1p-24h);
#else
  half x = __builtin_elementwise_canonicalize(0x1p-24h);
#endif
  return !__builtin_isfpclass(x, __FPCLASS_POSZERO);
}
#endif // cl_khr_fp16

_CLC_DEF bool __clc_fp32_subnormals_supported() {
#ifdef CLC_SPIRV
  float x = __clc_fabs(0x1p-149f);
#else
  float x = __builtin_elementwise_canonicalize(0x1p-149f);
#endif
  return !__builtin_isfpclass(x, __FPCLASS_POSZERO);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEF bool __clc_fp64_subnormals_supported() {
#ifdef CLC_SPIRV
  double x = __clc_fabs(0x1p-1074);
#else
  double x = __builtin_elementwise_canonicalize(0x1p-1074);
#endif
  return !__builtin_isfpclass(x, __FPCLASS_POSZERO);
}
#endif // cl_khr_fp64
