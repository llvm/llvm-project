//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) &&                                                   \
    ((defined(__CLC_FPSIZE) || __CLC_GENSIZE == 32 || __CLC_GENSIZE == 64) ||  \
     defined(cl_khr_subgroup_extended_types))

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE
sub_group_reduce_add(__CLC_GENTYPE x) {
  return __clc_sub_group_reduce_add(x);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE
sub_group_reduce_min(__CLC_GENTYPE x) {
  return __clc_sub_group_reduce_min(x);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE
sub_group_reduce_max(__CLC_GENTYPE x) {
  return __clc_sub_group_reduce_max(x);
}

#endif
