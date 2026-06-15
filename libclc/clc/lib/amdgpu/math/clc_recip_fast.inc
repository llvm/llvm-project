//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On AMDGPU the "fast" reciprocal is the hardware v_rcp_f32 approximation,
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_recip_fast(__CLC_GENTYPE x) {
#if defined(__CLC_SCALAR) && __CLC_FPSIZE == 32
  return __builtin_amdgcn_rcpf(x);
#else
  return ((__CLC_GENTYPE)1.0) / x;
#endif
}
