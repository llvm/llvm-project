//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/subgroup/clc_sub_group_broadcast.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_sub_group_reduce_add(uint x) {
  return __builtin_amdgcn_wave_reduce_add_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_reduce_add(int x) {
  return (int)__clc_sub_group_reduce_add((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong __clc_sub_group_reduce_add(ulong x) {
  return __builtin_amdgcn_wave_reduce_add_u64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long __clc_sub_group_reduce_add(long x) {
  return (long)__clc_sub_group_reduce_add((ulong)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_sub_group_reduce_min(uint x) {
  return __builtin_amdgcn_wave_reduce_min_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_reduce_min(int x) {
  return __builtin_amdgcn_wave_reduce_min_i32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong __clc_sub_group_reduce_min(ulong x) {
  return __builtin_amdgcn_wave_reduce_min_u64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long __clc_sub_group_reduce_min(long x) {
  return __builtin_amdgcn_wave_reduce_min_i64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint __clc_sub_group_reduce_max(uint x) {
  return __builtin_amdgcn_wave_reduce_max_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int __clc_sub_group_reduce_max(int x) {
  return __builtin_amdgcn_wave_reduce_max_i32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong __clc_sub_group_reduce_max(ulong x) {
  return __builtin_amdgcn_wave_reduce_max_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long __clc_sub_group_reduce_max(long x) {
  return __builtin_amdgcn_wave_reduce_max_i64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float __clc_sub_group_reduce_add(float x) {
  return __builtin_amdgcn_wave_reduce_fadd_f32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double __clc_sub_group_reduce_add(double x) {
  return __builtin_amdgcn_wave_reduce_fadd_f64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float __clc_sub_group_reduce_min(float x) {
  return __builtin_amdgcn_wave_reduce_fmin_f32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double __clc_sub_group_reduce_min(double x) {
  return __builtin_amdgcn_wave_reduce_fmin_f64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float __clc_sub_group_reduce_max(float x) {
  return __builtin_amdgcn_wave_reduce_fmax_f32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double __clc_sub_group_reduce_max(double x) {
  return __builtin_amdgcn_wave_reduce_fmax_f64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half __clc_sub_group_reduce_add(half x) {
  // FIXME: There should be a direct half builtin available.
  return (float)__clc_sub_group_reduce_add((float)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half __clc_sub_group_reduce_min(half x) {
  // FIXME: There should be a direct half builtin available.
  return (float)__clc_sub_group_reduce_min((float)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half __clc_sub_group_reduce_max(half x) {
  // FIXME: There should be a direct half builtin available.
  return (float)__clc_sub_group_reduce_max((float)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar __clc_sub_group_reduce_add(uchar x) {
  return (uchar)__clc_sub_group_reduce_add((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char __clc_sub_group_reduce_add(char x) {
  return (char)__clc_sub_group_reduce_add((int)x);
}

// FIXME: There should be a direct short builtin available.
_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort __clc_sub_group_reduce_add(ushort x) {
  return (ushort)__clc_sub_group_reduce_add((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short __clc_sub_group_reduce_add(short x) {
  return (int)__clc_sub_group_reduce_add((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar __clc_sub_group_reduce_min(uchar x) {
  return (uchar)__clc_sub_group_reduce_min((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char __clc_sub_group_reduce_min(char x) {
  return (char)__clc_sub_group_reduce_min((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort __clc_sub_group_reduce_min(ushort x) {
  return (ushort)__clc_sub_group_reduce_min((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short __clc_sub_group_reduce_min(short x) {
  return (int)__clc_sub_group_reduce_min((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar __clc_sub_group_reduce_max(uchar x) {
  return (uchar)__clc_sub_group_reduce_max((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char __clc_sub_group_reduce_max(char x) {
  return (char)__clc_sub_group_reduce_max((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort __clc_sub_group_reduce_max(ushort x) {
  return (ushort)__clc_sub_group_reduce_max((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short __clc_sub_group_reduce_max(short x) {
  return (int)__clc_sub_group_reduce_max((int)x);
}
