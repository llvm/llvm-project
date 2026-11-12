//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) && (defined(__CLC_FPSIZE) || __CLC_GENSIZE >= 32)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE work_group_reduce_add(__CLC_GENTYPE a) {
  return __clc_work_group_reduce_add(a);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE work_group_reduce_min(__CLC_GENTYPE a) {
  return __clc_work_group_reduce_min(a);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE work_group_reduce_max(__CLC_GENTYPE a) {
  return __clc_work_group_reduce_max(a);
}

#endif
