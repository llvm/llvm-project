//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR
_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_add(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_mul(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_min(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_max(__CLC_GENTYPE x);

#ifndef __CLC_FPSIZE
_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_and(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_or(__CLC_GENTYPE x);

_CLC_DECL _CLC_OVERLOAD __CLC_GENTYPE
__clc_sub_group_non_uniform_reduce_xor(__CLC_GENTYPE x);
#endif // __CLC_FPSIZE

#endif // __CLC_SCALAR
