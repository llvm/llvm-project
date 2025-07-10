//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/workitem/get_num_sub_groups.h>
#include <clc/workitem/clc_get_num_sub_groups.h>

_CLC_OVERLOAD _CLC_DEF uint get_num_sub_groups() {
  return __clc_get_num_sub_groups();
}
