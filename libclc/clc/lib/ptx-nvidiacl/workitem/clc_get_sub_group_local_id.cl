//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_sub_group_local_id.h>

_CLC_OVERLOAD _CLC_DEF uint __clc_get_sub_group_local_id() {
  return __nvvm_read_ptx_sreg_laneid();
}
