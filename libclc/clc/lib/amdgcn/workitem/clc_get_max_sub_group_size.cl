//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_max_sub_group_size.h>

_CLC_OVERLOAD _CLC_DEF uint __clc_get_max_sub_group_size() {
  return __builtin_amdgcn_wavefrontsize();
}
