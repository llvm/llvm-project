//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/opencl-base.h>

static uint wavesize_log2() {
  return __builtin_amdgcn_wavefrontsize() == 64 ? 6 : 5;
}

static uint workgroup_size() {
  return mul24((uint)get_local_size(2),
               mul24((uint)get_local_size(1), (uint)get_local_size(0)));
}

static uint enqueued_workgroup_size() {
  return mul24((uint)get_enqueued_local_size(2),
               mul24((uint)get_enqueued_local_size(1),
                     (uint)get_enqueued_local_size(0)));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_sub_group_size(void) {
  uint wavesize = __builtin_amdgcn_wavefrontsize();
  uint lid = (uint)get_local_linear_id();
  return min(wavesize, workgroup_size() - (lid & ~(wavesize - 1)));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_max_sub_group_size(void) {
  return min(__builtin_amdgcn_wavefrontsize(), enqueued_workgroup_size());
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_num_sub_groups(void) {
  return (workgroup_size() + __builtin_amdgcn_wavefrontsize() - 1) >>
         wavesize_log2();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_enqueued_num_sub_groups(void) {
  return (enqueued_workgroup_size() + __builtin_amdgcn_wavefrontsize() - 1) >>
         wavesize_log2();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_sub_group_id(void) {
  return (uint)get_local_linear_id() >> wavesize_log2();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint get_sub_group_local_id(void) {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int sub_group_all(int x) {
  return __builtin_amdgcn_ballot_w64(x) == __builtin_amdgcn_read_exec();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int sub_group_any(int x) {
  return __builtin_amdgcn_ballot_w64(x) != 0;
}
