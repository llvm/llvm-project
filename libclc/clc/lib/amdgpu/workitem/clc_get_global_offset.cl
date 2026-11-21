//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/workitem/clc_get_global_offset.h"
#include <amdhsa_abi.h>

_CLC_DEF _CLC_OVERLOAD size_t __clc_get_global_offset(uint dim) {
  __constant amdhsa_implicit_kernarg_v5 *implicit_args =
      (__constant amdhsa_implicit_kernarg_v5 *)
          __builtin_amdgcn_implicitarg_ptr();
  return dim < 3 ? implicit_args->global_offset[dim] : 0;
}
