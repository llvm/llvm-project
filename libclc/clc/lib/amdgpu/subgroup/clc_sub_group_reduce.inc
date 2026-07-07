//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_sub_group_reduce_add(__CLC_GENTYPE x) {
  return __clc_sub_group_non_uniform_reduce_add(x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_sub_group_reduce_min(__CLC_GENTYPE x) {
  return __clc_sub_group_non_uniform_reduce_min(x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_sub_group_reduce_max(__CLC_GENTYPE x) {
  return __clc_sub_group_non_uniform_reduce_max(x);
}

#endif
