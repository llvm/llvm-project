//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/subgroup/clc_sub_group_non_uniform_reduce.h"

#define __CLC_BODY "sub_group_non_uniform_reduce.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "sub_group_non_uniform_reduce.inc"
#include "clc/math/gentype.inc"

_CLC_DEF _CLC_OVERLOAD int
sub_group_non_uniform_reduce_logical_and(int predicate) {
  return __clc_sub_group_non_uniform_reduce_logical_and(predicate);
}

_CLC_DEF _CLC_OVERLOAD int
sub_group_non_uniform_reduce_logical_or(int predicate) {
  return __clc_sub_group_non_uniform_reduce_logical_or(predicate);
}

_CLC_DEF _CLC_OVERLOAD int
sub_group_non_uniform_reduce_logical_xor(int predicate) {
  return __clc_sub_group_non_uniform_reduce_logical_xor(predicate);
}
