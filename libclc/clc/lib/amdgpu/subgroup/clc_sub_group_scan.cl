//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/math/clc_fmax.h"
#include "clc/math/clc_fmin.h"
#include "clc/shared/clc_max.h"
#include "clc/shared/clc_min.h"
#include "clc/subgroup/clc_sub_group_broadcast.h"
#include "clc/subgroup/clc_sub_group_scan.h"
#include "clc/subgroup/clc_subgroup.h"

#define QUAD_PERM (1 << 15)

// The first basic swizzle mode (when offset[15] == 1) allows full data sharing
// between a group of 4 consecutive threads.
#define SWIZZLE_QUAD_PERM(S0, S1, S2, S3)                                      \
  (uint)(QUAD_PERM | (S3 << 6) | (S2 << 4) | (S1 << 2) | S0)

#define SWIZZLE_PAIRWISE(XOR_MASK, OR_MASK, AND_MASK)                          \
  (uint)((XOR_MASK << 10) | (OR_MASK << 5) | AND_MASK)

#define SWIZZLE_BCASTX2_LANE0 SWIZZLE_PAIRWISE(0x00, 0x00, 0x1e)
#define SWIZZLE_BCASTX4_LANE1 SWIZZLE_PAIRWISE(0x00, 0x01, 0x1c)
#define SWIZZLE_BCASTX8_LANE3 SWIZZLE_PAIRWISE(0x00, 0x03, 0x18)
#define SWIZZLE_BCASTX16_LANE7 SWIZZLE_PAIRWISE(0x00, 0x07, 0x10)
#define SWIZZLE_BCASTX32_LANE15 SWIZZLE_PAIRWISE(0x00, 0x0f, 0x00)

#define __CLC_BODY "clc_amdgpu_ds_swizzle.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc_amdgpu_ds_swizzle.inc"
#include "clc/math/gentype.inc"

//------------------------------------------------------------------------------
//  Integer and fp add
//------------------------------------------------------------------------------

#define __CLC_FUNCTION_INCLUSIVE __clc_sub_group_scan_inclusive_add
#define __CLC_FUNCTION_EXCLUSIVE __clc_sub_group_scan_exclusive_add
#define __CLC_FUNCTION_IMPL(x, y) ((x) + (y))
#define __CLC_SUBGROUP_SCAN_ID_VAL (__CLC_GENTYPE)0
#define __CLC_BODY "clc_sub_group_scan.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc_sub_group_scan.inc"
#include "clc/math/gentype.inc"

#undef __CLC_FUNCTION_INCLUSIVE
#undef __CLC_FUNCTION_EXCLUSIVE
#undef __CLC_FUNCTION_IMPL
#undef __CLC_SUBGROUP_SCAN_ID_VAL

//------------------------------------------------------------------------------
//  Integer and fp min
//------------------------------------------------------------------------------

#define __CLC_FUNCTION_INCLUSIVE __clc_sub_group_scan_inclusive_min
#define __CLC_FUNCTION_EXCLUSIVE __clc_sub_group_scan_exclusive_min
#define __CLC_FUNCTION_IMPL(x, y) __clc_min(x, y)
#define __CLC_SUBGROUP_SCAN_ID_VAL __CLC_GEN_MAX
#define __CLC_BODY "clc_sub_group_scan.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc_sub_group_scan.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION_IMPL
#undef __CLC_FUNCTION_INCLUSIVE
#undef __CLC_FUNCTION_EXCLUSIVE
#undef __CLC_SUBGROUP_SCAN_ID_VAL

//------------------------------------------------------------------------------
//  Integer and fp max
//------------------------------------------------------------------------------

#define __CLC_FUNCTION_INCLUSIVE __clc_sub_group_scan_inclusive_max
#define __CLC_FUNCTION_EXCLUSIVE __clc_sub_group_scan_exclusive_max
#define __CLC_FUNCTION_IMPL(x, y) __clc_max(x, y)
#define __CLC_SUBGROUP_SCAN_ID_VAL __CLC_GEN_MIN

#define __CLC_BODY "clc_sub_group_scan.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "clc_sub_group_scan.inc"
#include "clc/math/gentype.inc"
#undef __CLC_FUNCTION_IMPL
#undef __CLC_FUNCTION_INCLUSIVE
#undef __CLC_FUNCTION_EXCLUSIVE
#undef __CLC_SUBGROUP_SCAN_ID_VAL
