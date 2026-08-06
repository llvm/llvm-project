//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/atomic/clc_atomic_fetch_and.h"
#include "clc/atomic/clc_atomic_fetch_or.h"
#include "clc/atomic/clc_atomic_load.h"
#include "clc/atomic/clc_atomic_store.h"
#include "clc/collective/clc_work_group_any_all.h"
#include "clc/subgroup/clc_subgroup.h"
#include "clc/synchronization/clc_work_group_barrier.h"

#pragma OPENCL EXTENSION __cl_clang_function_scope_local_variables : enable

static int work_group_any_all_impl(int predicate, bool is_all) {
  __local uint scratch;

  uint n = __clc_get_num_sub_groups();
  int a =
      is_all ? __clc_sub_group_all(predicate) : __clc_sub_group_any(predicate);
  if (n == 1)
    return a;

  uint l = __clc_get_sub_group_local_id();
  uint i = __clc_get_sub_group_id();

  if ((i == 0) & (l == 0))
    __clc_atomic_store(&scratch, a, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);

  if ((i != 0) & (l == 0)) {
    if (is_all)
      __clc_atomic_fetch_and(&scratch, a, __ATOMIC_RELAXED,
                             __MEMORY_SCOPE_WRKGRP);
    else
      __clc_atomic_fetch_or(&scratch, a, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_WRKGRP);
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  a = __clc_atomic_load(&scratch, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);

  return a;
}

_CLC_OVERLOAD _CLC_DEF int __clc_work_group_all(int predicate) {
  return work_group_any_all_impl(predicate, true);
}

_CLC_OVERLOAD _CLC_DEF int __clc_work_group_any(int predicate) {
  return work_group_any_all_impl(predicate, false);
}
