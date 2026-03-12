//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <amdhsa_abi.h>
#include <clc/workitem/clc_get_work_dim.h>

_CLC_OVERLOAD _CLC_DEF uint __clc_get_work_dim() {
  __constant amdhsa_implicit_kernarg_v5 *implicit_args =
      (__constant amdhsa_implicit_kernarg_v5 *)
          __builtin_amdgcn_implicitarg_ptr();
  return implicit_args->grid_dims;
}
