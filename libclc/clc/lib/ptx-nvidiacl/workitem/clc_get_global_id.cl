//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_global_id.h>
#include <clc/workitem/clc_get_group_id.h>
#include <clc/workitem/clc_get_local_id.h>
#include <clc/workitem/clc_get_local_size.h>

_CLC_OVERLOAD _CLC_DEF size_t __clc_get_global_id(uint dim) {
  return __clc_get_group_id(dim) * __clc_get_local_size(dim) +
         __clc_get_local_id(dim);
}
