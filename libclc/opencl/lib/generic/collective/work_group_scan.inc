//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_SCALAR) && (defined(__CLC_FPSIZE) || __CLC_GENSIZE >= 32)

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
work_group_scan_inclusive_add(__CLC_GENTYPE a) {
  return __clc_work_group_scan_inclusive_add(a);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
work_group_scan_inclusive_min(__CLC_GENTYPE a) {
  return __clc_work_group_scan_inclusive_min(a);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
work_group_scan_inclusive_max(__CLC_GENTYPE a) {
  return __clc_work_group_scan_inclusive_max(a);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
work_group_scan_exclusive_add(__CLC_GENTYPE a) {
  return __clc_work_group_scan_exclusive_add(a);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
work_group_scan_exclusive_min(__CLC_GENTYPE a) {
  return __clc_work_group_scan_exclusive_min(a);
}

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE
work_group_scan_exclusive_max(__CLC_GENTYPE a) {
  return __clc_work_group_scan_exclusive_max(a);
}

#endif
