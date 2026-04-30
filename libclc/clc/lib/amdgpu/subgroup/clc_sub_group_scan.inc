//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR)

#if defined(__CLC_GEN_S)
#define __CLC_GEN_MAX (__CLC_GENTYPE)((1LL << (__CLC_GENSIZE - 1)) - 1LL)
#define __CLC_GEN_MIN (__CLC_GENTYPE)(-(1LL << (__CLC_GENSIZE - 1)))
#elif defined(__CLC_GEN_U)
#define __CLC_GEN_MAX (__CLC_GENTYPE)((1ull << __CLC_GENSIZE) - 1ull)
#define __CLC_GEN_MIN (__CLC_GENTYPE)0
#elif defined(__CLC_FPSIZE)
#define __CLC_GEN_MIN -INFINITY
#define __CLC_GEN_MAX INFINITY
#endif

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __CLC_FUNCTION_INCLUSIVE(__CLC_GENTYPE x) {
  uint l = __clc_get_sub_group_local_id();

  __CLC_GENTYPE v = __clc_amdgpu_ds_swizzle_bcastx2_lane0(x);
  v = (l & 1) ? v : __CLC_SUBGROUP_SCAN_ID_VAL;
  __CLC_GENTYPE s = __CLC_FUNCTION_IMPL(x, v);

  v = __clc_amdgpu_ds_swizzle_bcastx4_lane1(s);
  v = (l & 2) ? v : __CLC_SUBGROUP_SCAN_ID_VAL;
  s = __CLC_FUNCTION_IMPL(s, v);

  v = __clc_amdgpu_ds_swizzle_bcastx8_lane3(s);
  v = (l & 4) ? v : __CLC_SUBGROUP_SCAN_ID_VAL;
  s = __CLC_FUNCTION_IMPL(s, v);

  v = __clc_amdgpu_ds_swizzle_bcastx16_lane7(s);
  v = (l & 8) ? v : __CLC_SUBGROUP_SCAN_ID_VAL;
  s = __CLC_FUNCTION_IMPL(s, v);

  v = __clc_amdgpu_ds_swizzle_bcastx32_lane15(s);
  v = (l & 16) ? v : __CLC_SUBGROUP_SCAN_ID_VAL;
  s = __CLC_FUNCTION_IMPL(s, v);

  if (__builtin_amdgcn_wavefrontsize() == 64) {
    v = __clc_sub_group_broadcast(s, 31);
    v = l > 31 ? v : __CLC_SUBGROUP_SCAN_ID_VAL;
    s = __CLC_FUNCTION_IMPL(s, v);
  }

  return s;
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __CLC_FUNCTION_EXCLUSIVE(__CLC_GENTYPE x) {
  __CLC_GENTYPE s = __CLC_FUNCTION_INCLUSIVE(x);
  __CLC_GENTYPE t = s;

  s = __clc_amdgpu_ds_swizzle_quad_perm_shift_right1(t);

  __CLC_GENTYPE v = __clc_amdgpu_ds_swizzle_bcastx8_lane3(t);

  uint l = __clc_get_sub_group_local_id();

  s = ((l & 0x7) == 0x4) ? v : s;

  v = __clc_amdgpu_ds_swizzle_bcastx16_lane7(t);
  s = ((l & 0xf) == 0x8) ? v : s;

  v = __clc_amdgpu_ds_swizzle_bcastx32_lane15(t);
  s = ((l & 0x1f) == 0x10) ? v : s;

  if (__builtin_amdgcn_wavefrontsize() == 64) {
    v = __clc_sub_group_broadcast(t, 31);
    s = (l == 32) ? v : s;
  }

  return (l == 0) ? __CLC_SUBGROUP_SCAN_ID_VAL : s;
}

#undef __CLC_GEN_MIN
#undef __CLC_GEN_MAX

#endif // __CLC_SCALAR
