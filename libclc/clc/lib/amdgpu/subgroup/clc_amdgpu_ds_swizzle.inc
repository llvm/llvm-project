//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR)

#if (defined(__CLC_GENSIZE) && __CLC_GENSIZE <= 32) ||                         \
    defined(__CLC_FPSIZE) && __CLC_FPSIZE <= 32

#define COERCE_FUNC(func, mask)                                                \
  static _CLC_OVERLOAD __CLC_GENTYPE func(__CLC_GENTYPE x) {                   \
    __CLC_U_GENTYPE bitcast = __CLC_AS_U_GENTYPE(x);                           \
    uint ext = __clc_convert_uint(bitcast);                                    \
    uint swizzle = __builtin_amdgcn_ds_swizzle(ext, mask);                     \
    __CLC_U_GENTYPE trunc = __CLC_CONVERT_U_GENTYPE(swizzle);                  \
    return __CLC_AS_GENTYPE(trunc);                                            \
  }

//------------------------------------------------------------------------------
// Swizzle masks used in inclusive scan
//------------------------------------------------------------------------------

COERCE_FUNC(__clc_amdgpu_ds_swizzle_bcastx2_lane0, SWIZZLE_BCASTX2_LANE0);
COERCE_FUNC(__clc_amdgpu_ds_swizzle_bcastx4_lane1, SWIZZLE_BCASTX4_LANE1);
COERCE_FUNC(__clc_amdgpu_ds_swizzle_bcastx8_lane3, SWIZZLE_BCASTX8_LANE3);
COERCE_FUNC(__clc_amdgpu_ds_swizzle_bcastx16_lane7, SWIZZLE_BCASTX16_LANE7);
COERCE_FUNC(__clc_amdgpu_ds_swizzle_bcastx32_lane15, SWIZZLE_BCASTX32_LANE15);

//------------------------------------------------------------------------------
// Swizzle masks used in exclusive scan adjustment
//------------------------------------------------------------------------------

COERCE_FUNC(__clc_amdgpu_ds_swizzle_quad_perm_shift_right1,
            SWIZZLE_QUAD_PERM(0, 0, 1, 2));

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
