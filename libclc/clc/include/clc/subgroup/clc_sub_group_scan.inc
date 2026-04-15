//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR
_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_scan_inclusive_add(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_scan_inclusive_min(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_scan_inclusive_max(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_scan_exclusive_add(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_scan_exclusive_min(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_scan_exclusive_max(__CLC_GENTYPE x);
#endif
