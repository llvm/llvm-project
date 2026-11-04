//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) &&                                                   \
    (defined(__CLC_FPSIZE) || __CLC_GENSIZE == 32 || __CLC_GENSIZE == 64)

static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE sub_group_scan_op(
    __CLC_GENTYPE x, enum __CLC_WORK_GROUP_SCAN_OP opcode, bool inclusive) {
  switch (opcode) {
  case __CLC_WORK_GROUP_SCAN_ADD:
    return inclusive ? __clc_sub_group_scan_inclusive_add(x)
                     : __clc_sub_group_scan_exclusive_add(x);
  case __CLC_WORK_GROUP_SCAN_MIN:
    return inclusive ? __clc_sub_group_scan_inclusive_min(x)
                     : __clc_sub_group_scan_exclusive_min(x);
  case __CLC_WORK_GROUP_SCAN_MAX:
    return inclusive ? __clc_sub_group_scan_inclusive_max(x)
                     : __clc_sub_group_scan_exclusive_max(x);
  }
}

static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE scan_op(
    __CLC_GENTYPE x, __CLC_GENTYPE y, enum __CLC_WORK_GROUP_SCAN_OP opcode) {
  switch (opcode) {
  case __CLC_WORK_GROUP_SCAN_ADD:
    return x + y;
  case __CLC_WORK_GROUP_SCAN_MIN:
    return __clc_min(x, y);
  case __CLC_WORK_GROUP_SCAN_MAX:
    return __clc_max(x, y);
  }
}

#ifdef __CLC_FPSIZE
static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE work_group_scan_identity_value(
    __CLC_GENTYPE x, enum __CLC_WORK_GROUP_SCAN_OP opcode) {
  (void)x;
  switch (opcode) {
  case __CLC_WORK_GROUP_SCAN_ADD:
    return __CLC_FP_LIT(0.0);
  case __CLC_WORK_GROUP_SCAN_MIN:
    return (__CLC_GENTYPE)INFINITY;
  case __CLC_WORK_GROUP_SCAN_MAX:
    return (__CLC_GENTYPE)-INFINITY;
  }
}
#else
static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE work_group_scan_identity_value(
    __CLC_GENTYPE x, enum __CLC_WORK_GROUP_SCAN_OP opcode) {
  (void)x;
  switch (opcode) {
  case __CLC_WORK_GROUP_SCAN_ADD:
    return (__CLC_GENTYPE)0;
  case __CLC_WORK_GROUP_SCAN_MIN:
#ifdef __CLC_GEN_S
    return (__CLC_GENTYPE)LONG_MAX;
#else
    return (__CLC_GENTYPE)ULONG_MAX;
#endif
  case __CLC_WORK_GROUP_SCAN_MAX:
#ifdef __CLC_GEN_S
    return (__CLC_GENTYPE)LONG_MIN;
#else
    return (__CLC_GENTYPE)0;
#endif
  }
}
#endif

static _CLC_OVERLOAD __CLC_GENTYPE __clc_work_group_scan_impl(
    __CLC_GENTYPE a, enum __CLC_WORK_GROUP_SCAN_OP opcode, bool inclusive) {
  uint n = __clc_get_num_sub_groups();
  __CLC_GENTYPE t = sub_group_scan_op(a, opcode, inclusive);
  if (n == 1)
    return t;

  __local __CLC_GENTYPE scratch[__CLC_MAX_NUM_WORK_GROUPS];
  uint l = __clc_get_sub_group_local_id();
  uint i = __clc_get_sub_group_id();

  if (l == __clc_get_sub_group_size() - 1u) {
    __CLC_GENTYPE store_val = inclusive ? t : scan_op(a, t, opcode);
    __scoped_atomic_store_n(&scratch[i], store_val, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_WRKGRP);
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);

  if (i == 0) {
    __CLC_GENTYPE s =
        l < n ? __scoped_atomic_load_n(&scratch[l], __ATOMIC_RELAXED,
                                       __MEMORY_SCOPE_WRKGRP)
              : work_group_scan_identity_value(t, opcode);
    s = sub_group_scan_op(s, opcode, /*inclusive=*/true);
    if (l < n) {
      __scoped_atomic_store_n(&scratch[l], s, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_WRKGRP);
    }
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);

  __CLC_GENTYPE ret = t;
  if (i != 0) {
    __CLC_GENTYPE load_scratch = __scoped_atomic_load_n(
        &scratch[i - 1], __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    ret = scan_op(t, load_scratch, opcode);
  }

  __clc_work_group_barrier(__MEMORY_SCOPE_WRKGRP, __CLC_MEMORY_LOCAL);
  return ret;
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_work_group_scan_inclusive_add(__CLC_GENTYPE a) {
  return __clc_work_group_scan_impl(a, __CLC_WORK_GROUP_SCAN_ADD,
                                    /*inclusive=*/true);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_work_group_scan_inclusive_min(__CLC_GENTYPE a) {
  return __clc_work_group_scan_impl(a, __CLC_WORK_GROUP_SCAN_MIN,
                                    /*inclusive=*/true);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_work_group_scan_inclusive_max(__CLC_GENTYPE a) {
  return __clc_work_group_scan_impl(a, __CLC_WORK_GROUP_SCAN_MAX,
                                    /*inclusive=*/true);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_work_group_scan_exclusive_add(__CLC_GENTYPE a) {
  return __clc_work_group_scan_impl(a, __CLC_WORK_GROUP_SCAN_ADD,
                                    /*inclusive=*/false);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_work_group_scan_exclusive_min(__CLC_GENTYPE a) {
  return __clc_work_group_scan_impl(a, __CLC_WORK_GROUP_SCAN_MIN,
                                    /*inclusive=*/false);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
__clc_work_group_scan_exclusive_max(__CLC_GENTYPE a) {
  return __clc_work_group_scan_impl(a, __CLC_WORK_GROUP_SCAN_MAX,
                                    /*inclusive=*/false);
}

#endif
