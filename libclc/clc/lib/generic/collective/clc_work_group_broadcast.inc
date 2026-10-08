//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

_CLC_OVERLOAD
static __CLC_GENTYPE __clc_work_group_broadcast_impl(__CLC_GENTYPE a,
                                                     bool is_leader) {
  __local __CLC_GENTYPE scratch;
  if (is_leader) {
    __scoped_atomic_store_n(&scratch, a, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_WRKGRP);
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  __CLC_GENTYPE result =
      __scoped_atomic_load_n(&scratch, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  return result;
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_work_group_broadcast(__CLC_GENTYPE a, size_t local_id_x) {
  if (__clc_get_num_sub_groups() == 1)
    return __clc_sub_group_broadcast(a, local_id_x);

  bool is_leader = __clc_get_local_id(0) == local_id_x;
  return __clc_work_group_broadcast_impl(a, is_leader);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_work_group_broadcast(
    __CLC_GENTYPE a, size_t local_id_x, size_t local_id_y) {
  bool is_leader = __clc_get_local_id(0) == local_id_x &&
                   __clc_get_local_id(1) == local_id_y;
  return __clc_work_group_broadcast_impl(a, is_leader);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_work_group_broadcast(
    __CLC_GENTYPE a, size_t local_id_x, size_t local_id_y, size_t local_id_z) {
  bool is_leader = __clc_get_local_id(0) == local_id_x &&
                   __clc_get_local_id(1) == local_id_y &&
                   __clc_get_local_id(2) == local_id_z;
  return __clc_work_group_broadcast_impl(a, is_leader);
}

#endif // __CLC_SCALAR
