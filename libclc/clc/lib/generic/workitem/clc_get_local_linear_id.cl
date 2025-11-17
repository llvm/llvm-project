//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_local_id.h>
#include <clc/workitem/clc_get_local_linear_id.h>
#include <clc/workitem/clc_get_local_size.h>

_CLC_OVERLOAD _CLC_DEF size_t __clc_get_local_linear_id() {
  return __clc_get_local_id(2) * __clc_get_local_size(1) *
             __clc_get_local_size(0) +
         __clc_get_local_id(1) * __clc_get_local_size(0) +
         __clc_get_local_id(0);
}
