//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On AMDGPU the "fast" reciprocal is the hardware v_rcp approximation.
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_recip_fast(__CLC_GENTYPE x) {
#if defined(__CLC_SCALAR) && __CLC_FPSIZE == 32
  return __builtin_amdgcn_rcpf(x);
#elif defined(__CLC_SCALAR) && __CLC_FPSIZE == 64
  // Hardware v_rcp_f64 seed refined with two Newton-Raphson iterations. This
  // computes 1.0 / x without the full IEEE scaling and subnormal fixups.
  __CLC_GENTYPE r = __builtin_amdgcn_rcp(x);
  r = __builtin_fma(__builtin_fma(-x, r, 1.0), r, r);
  r = __builtin_fma(__builtin_fma(-x, r, 1.0), r, r);
  return r;
#else
  return ((__CLC_GENTYPE)1.0) / x;
#endif
}
