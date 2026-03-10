//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR)

#if (defined(__CLC_GEN_S) && __CLC_GENSIZE == 32) ||                           \
    defined(__CLC_FPSIZE) && __CLC_FPSIZE == 32

static _CLC_OVERLOAD __CLC_GENTYPE
__clc_amdgpu_ds_swizzle_bcastx2_lane0(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(
      __clc_amdgpu_ds_swizzle_bcastx2_lane0(__clc_as_uint(x)));
}

static _CLC_OVERLOAD __CLC_GENTYPE
__clc_amdgpu_ds_swizzle_bcastx4_lane1(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(
      __clc_amdgpu_ds_swizzle_bcastx4_lane1(__clc_as_uint(x)));
}

static _CLC_OVERLOAD __CLC_GENTYPE
__clc_amdgpu_ds_swizzle_bcastx8_lane3(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(
      __clc_amdgpu_ds_swizzle_bcastx8_lane3(__clc_as_uint(x)));
}

static _CLC_OVERLOAD __CLC_GENTYPE
__clc_amdgpu_ds_swizzle_bcastx16_lane7(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(
      __clc_amdgpu_ds_swizzle_bcastx16_lane7(__clc_as_uint(x)));
}

static _CLC_OVERLOAD __CLC_GENTYPE
__clc_amdgpu_ds_swizzle_bcastx32_lane15(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(
      __clc_amdgpu_ds_swizzle_bcastx32_lane15(__clc_as_uint(x)));
}

_CLC_OVERLOAD static __CLC_GENTYPE
__clc_amdgpu_ds_swizzle_quad_perm_shift_right1(__CLC_GENTYPE x) {
  return __CLC_AS_GENTYPE(
      __clc_amdgpu_ds_swizzle_quad_perm_shift_right1(__clc_as_uint(x)));
}

#elif defined(__CLC_GENSIZE) && __CLC_GENSIZE < 32 ||                          \
    defined(__CLC_FPSIZE) && __CLC_FPSIZE < 32

#define PROMOTE_FUNC_UINT(func_name)                                           \
  static _CLC_OVERLOAD __CLC_GENTYPE func_name(__CLC_GENTYPE x) {              \
    return __CLC_AS_GENTYPE(__CLC_CONVERT_U_GENTYPE(                           \
        func_name(__clc_convert_uint(__CLC_AS_U_GENTYPE(x)))));                \
  }

PROMOTE_FUNC_UINT(__clc_amdgpu_ds_swizzle_bcastx2_lane0)
PROMOTE_FUNC_UINT(__clc_amdgpu_ds_swizzle_bcastx4_lane1)
PROMOTE_FUNC_UINT(__clc_amdgpu_ds_swizzle_bcastx8_lane3)
PROMOTE_FUNC_UINT(__clc_amdgpu_ds_swizzle_bcastx16_lane7)
PROMOTE_FUNC_UINT(__clc_amdgpu_ds_swizzle_bcastx32_lane15)

PROMOTE_FUNC_UINT(__clc_amdgpu_ds_swizzle_quad_perm_shift_right1)

#elif defined(__CLC_GENSIZE) && __CLC_GENSIZE == 64 ||                         \
    defined(__CLC_FPSIZE) && __CLC_FPSIZE == 64

#define SPLIT_FUNC_64(func_name)                                               \
  static _CLC_OVERLOAD __CLC_GENTYPE func_name(__CLC_GENTYPE x) {              \
    uint2 vec = __clc_as_uint2(x);                                             \
    uint2 r = {func_name(vec.lo), func_name(vec.hi)};                          \
    return __CLC_AS_GENTYPE(r);                                                \
  }

SPLIT_FUNC_64(__clc_amdgpu_ds_swizzle_bcastx2_lane0)
SPLIT_FUNC_64(__clc_amdgpu_ds_swizzle_bcastx4_lane1)
SPLIT_FUNC_64(__clc_amdgpu_ds_swizzle_bcastx8_lane3)
SPLIT_FUNC_64(__clc_amdgpu_ds_swizzle_bcastx16_lane7)
SPLIT_FUNC_64(__clc_amdgpu_ds_swizzle_bcastx32_lane15)

SPLIT_FUNC_64(__clc_amdgpu_ds_swizzle_quad_perm_shift_right1)

#endif

#endif // __CLC_SCALAR
