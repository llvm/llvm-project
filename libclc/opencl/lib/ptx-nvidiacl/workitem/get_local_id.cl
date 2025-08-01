//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/workitem/get_local_id.h>
#include <clc/workitem/clc_get_local_id.h>

_CLC_OVERLOAD _CLC_DEF size_t get_local_id(uint dim) {
  return __clc_get_local_id(dim);
}
