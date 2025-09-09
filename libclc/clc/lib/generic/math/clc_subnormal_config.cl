//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_subnormal_config.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEF bool __clc_fp16_subnormals_supported() {
#ifdef CLC_SPIRV
  // SPIR-V doesn't support llvm.canonicalize for now.
  return false;
#else
  return !__builtin_isfpclass(__builtin_canonicalizef((float)0x1p-24h),
                              __FPCLASS_POSZERO);
#endif
}
#endif // cl_khr_fp16

_CLC_DEF bool __clc_fp32_subnormals_supported() {
#ifdef CLC_SPIRV
  // SPIR-V doesn't support llvm.canonicalize for now.
  return false;
#else
  return !__builtin_isfpclass(__builtin_canonicalizef(0x1p-149f),
                              __FPCLASS_POSZERO);
#endif
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEF bool __clc_fp64_subnormals_supported() {
#ifdef CLC_SPIRV
  // SPIR-V doesn't support llvm.canonicalize for now.
  return false;
#else
  return !__builtin_isfpclass(__builtin_canonicalize(0x1p-1074),
                              __FPCLASS_POSZERO);
#endif
}
#endif // cl_khr_fp64
