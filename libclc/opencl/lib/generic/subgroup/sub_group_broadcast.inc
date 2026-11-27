//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) || defined(cl_khr_subgroup_extended_types)

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE
sub_group_broadcast(__CLC_GENTYPE x, uint sub_group_local_id) {
  return __clc_sub_group_broadcast(x, sub_group_local_id);
}

#endif
