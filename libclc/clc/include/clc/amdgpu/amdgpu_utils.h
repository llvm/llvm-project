//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/integer/clc_mul24.h"
#include "clc/workitem/clc_get_enqueued_local_size.h"
#include "clc/workitem/clc_get_local_size.h"

static inline uint __clc_amdgpu_workgroup_size() {
  return __clc_mul24((uint)__clc_get_local_size(2),
                     __clc_mul24((uint)__clc_get_local_size(1),
                                 (uint)__clc_get_local_size(0)));
}

static inline uint __clc_amdgpu_enqueued_workgroup_size() {
  return __clc_mul24((uint)__clc_get_enqueued_local_size(2),
                     __clc_mul24((uint)__clc_get_enqueued_local_size(1),
                                 (uint)__clc_get_enqueued_local_size(0)));
}

static inline uint __clc_amdgpu_wavesize_log2() {
  return __builtin_amdgcn_wavefrontsize() == 64 ? 6 : 5;
}
