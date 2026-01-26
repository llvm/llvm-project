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
  // SPIR-V doesn't support llvm.canonicalize. Synthesize a subnormal by halving
  // the smallest normal. If subnormals are not supported it will flush to +0.
  half smallest_normal = 0x1p-14h;
  half sub =
      smallest_normal * 0.5h; // Expected 0x1p-15h (subnormal) if supported
  return !__builtin_isfpclass(sub, __FPCLASS_POSZERO);
}
#endif // cl_khr_fp16

_CLC_DEF bool __clc_fp32_subnormals_supported() {
  // SPIR-V doesn't support llvm.canonicalize. Synthesize a subnormal by halving
  // the smallest normal. If subnormals are not supported it will flush to +0.
  float smallest_normal = 0x1p-126f;
  float sub =
      smallest_normal * 0.5f; // Should be 0x1p-127f (subnormal) if supported
  return !__builtin_isfpclass(sub, __FPCLASS_POSZERO);
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEF bool __clc_fp64_subnormals_supported() {
  // SPIR-V doesn't support llvm.canonicalize. Synthesize a subnormal by halving
  // the smallest normal. If subnormals are not supported it will flush to +0.
  double smallest_normal = 0x1p-1022;
  double sub =
      smallest_normal * 0.5; // Should be 0x1p-1023 (subnormal) if supported
  return !__builtin_isfpclass(sub, __FPCLASS_POSZERO);
}
#endif // cl_khr_fp64
