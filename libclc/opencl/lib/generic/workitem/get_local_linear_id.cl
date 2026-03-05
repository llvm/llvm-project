//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/opencl-base.h>

_CLC_OVERLOAD _CLC_DEF _CLC_CONST size_t get_local_linear_id() {
  return (get_local_id(2) * get_local_size(1) + get_local_id(1)) *
             get_local_size(0) +
         get_local_id(0);
}
