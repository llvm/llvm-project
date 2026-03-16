//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) &&                                                   \
    (defined(__CLC_FPSIZE) || __CLC_GENSIZE == 32 || __CLC_GENSIZE == 64)

_CLC_OVERLOAD _CLC_CONST static __CLC_GENTYPE
sub_group_reduce_op(__CLC_GENTYPE x, enum __CLC_WORK_GROUP_REDUCE_OP opcode) {
  switch (opcode) {
  case __CLC_WORK_GROUP_REDUCE_ADD:
    return __clc_sub_group_reduce_add(x);
  case __CLC_WORK_GROUP_REDUCE_MIN:
    return __clc_sub_group_reduce_min(x);
  case __CLC_WORK_GROUP_REDUCE_MAX:
    return __clc_sub_group_reduce_max(x);
  }
}

_CLC_OVERLOAD
static __CLC_GENTYPE atomic_reduce_op(__local __CLC_GENTYPE *ptr,
                                      __CLC_GENTYPE x,
                                      enum __CLC_WORK_GROUP_REDUCE_OP opcode) {
  switch (opcode) {
  case __CLC_WORK_GROUP_REDUCE_ADD:
    return __scoped_atomic_fetch_add(ptr, x, __ATOMIC_RELAXED,
                                     __MEMORY_SCOPE_WRKGRP);
  case __CLC_WORK_GROUP_REDUCE_MIN:
    return __scoped_atomic_fetch_min(ptr, x, __ATOMIC_RELAXED,
                                     __MEMORY_SCOPE_WRKGRP);
  case __CLC_WORK_GROUP_REDUCE_MAX:
    return __scoped_atomic_fetch_max(ptr, x, __ATOMIC_RELAXED,
                                     __MEMORY_SCOPE_WRKGRP);
  }
}

_CLC_OVERLOAD
static __CLC_GENTYPE
__clc_work_group_reduce_impl(__CLC_GENTYPE a,
                             enum __CLC_WORK_GROUP_REDUCE_OP opcode) {
  uint n = __clc_get_num_sub_groups();
  a = sub_group_reduce_op(a, opcode);
  if (n == 1)
    return a;

  __local __CLC_GENTYPE scratch;
  uint l = __clc_get_sub_group_local_id();
  uint i = __clc_get_sub_group_id();

  if ((i == 0) & (l == 0)) {
    __scoped_atomic_store_n(&scratch, a, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_WRKGRP);
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  if ((i != 0) & (l == 0)) {
    atomic_reduce_op(&scratch, a, opcode);
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  __CLC_GENTYPE ret =
      __scoped_atomic_load_n(&scratch, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  return ret;
}

_CLC_OVERLOAD
__CLC_GENTYPE __clc_work_group_reduce_add(__CLC_GENTYPE a) {
  return __clc_work_group_reduce_impl(a, __CLC_WORK_GROUP_REDUCE_ADD);
}

_CLC_OVERLOAD
__CLC_GENTYPE __clc_work_group_reduce_min(__CLC_GENTYPE a) {
  return __clc_work_group_reduce_impl(a, __CLC_WORK_GROUP_REDUCE_MIN);
}

_CLC_OVERLOAD
__CLC_GENTYPE __clc_work_group_reduce_max(__CLC_GENTYPE a) {
  return __clc_work_group_reduce_impl(a, __CLC_WORK_GROUP_REDUCE_MAX);
}

#endif
