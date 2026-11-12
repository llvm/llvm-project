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

#if __has_builtin(__builtin_amdgcn_update_dpp)

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __CLC_FUNCTION_INCLUSIVE(__CLC_GENTYPE x) {
  __CLC_GENTYPE id = (__CLC_GENTYPE)__CLC_SUBGROUP_SCAN_ID_VAL;

  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, x, DPP_ROW_SHR | 1, 0xf, 0xf, 0));
  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, x, DPP_ROW_SHR | 2, 0xf, 0xf, 0));
  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, x, DPP_ROW_SHR | 4, 0xf, 0xf, 0));
  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, x, DPP_ROW_SHR | 8, 0xf, 0xf, 0));

#if __has_builtin(__builtin_amdgcn_permlanex16)
  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, __clc_amdgpu_permlanex16(id, x, -1, -1),
                                 DPP_ID_PERM, 0xa, 0xf, 0));
  if (__builtin_amdgcn_wavefrontsize() == 64) {
    __CLC_GENTYPE lane31 = __clc_sub_group_broadcast(x, 31);
    x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                   id, lane31, DPP_ID_PERM, 0xc, 0xf, 0));
  }
#else
  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, x, DPP_ROW_BCAST15, 0xa, 0xf, 0));
  x = __CLC_FUNCTION_IMPL(x, (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
                                 id, x, DPP_ROW_BCAST31, 0xc, 0xf, 0));
#endif

  return x;
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __CLC_FUNCTION_EXCLUSIVE(__CLC_GENTYPE x) {
  __CLC_GENTYPE s = __CLC_FUNCTION_INCLUSIVE(x);
  __CLC_GENTYPE id = (__CLC_GENTYPE)__CLC_SUBGROUP_SCAN_ID_VAL;

  __CLC_GENTYPE r = (__CLC_GENTYPE)__builtin_amdgcn_update_dpp(
      id, s, DPP_ROW_SHR | 1, 0xf, 0xf, 0);

  uint l = __clc_get_sub_group_local_id();
  r = (l == 16) ? __clc_sub_group_broadcast(s, 15) : r;
  if (__builtin_amdgcn_wavefrontsize() == 64) {
    r = (l == 32) ? __clc_sub_group_broadcast(s, 31) : r;
    r = (l == 48) ? __clc_sub_group_broadcast(s, 47) : r;
  }

  return r;
}

#else // !__has_builtin(__builtin_amdgcn_update_dpp)

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

#endif // __has_builtin(__builtin_amdgcn_update_dpp)

#undef __CLC_GEN_MIN
#undef __CLC_GEN_MAX

#endif // __CLC_SCALAR
