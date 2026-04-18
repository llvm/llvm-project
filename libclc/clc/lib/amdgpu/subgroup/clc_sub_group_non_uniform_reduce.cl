//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/subgroup/clc_sub_group_non_uniform_reduce.h"

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_add(uint x) {
  return __builtin_amdgcn_wave_reduce_add_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_add(int x) {
  return (int)__clc_sub_group_non_uniform_reduce_add((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_add(ulong x) {
  return __builtin_amdgcn_wave_reduce_add_u64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_add(long x) {
  return (long)__clc_sub_group_non_uniform_reduce_add((ulong)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_min(uint x) {
  return __builtin_amdgcn_wave_reduce_min_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_min(int x) {
  return __builtin_amdgcn_wave_reduce_min_i32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_min(ulong x) {
  return __builtin_amdgcn_wave_reduce_min_u64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_min(long x) {
  return __builtin_amdgcn_wave_reduce_min_i64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_max(uint x) {
  return __builtin_amdgcn_wave_reduce_max_u32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_max(int x) {
  return __builtin_amdgcn_wave_reduce_max_i32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_max(ulong x) {
  return __builtin_amdgcn_wave_reduce_max_u64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_max(long x) {
  return __builtin_amdgcn_wave_reduce_max_i64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float
__clc_sub_group_non_uniform_reduce_add(float x) {
  return __builtin_amdgcn_wave_reduce_fadd_f32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double
__clc_sub_group_non_uniform_reduce_add(double x) {
  return __builtin_amdgcn_wave_reduce_fadd_f64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float
__clc_sub_group_non_uniform_reduce_min(float x) {
  return __builtin_amdgcn_wave_reduce_fmin_f32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double
__clc_sub_group_non_uniform_reduce_min(double x) {
  return __builtin_amdgcn_wave_reduce_fmin_f64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float
__clc_sub_group_non_uniform_reduce_max(float x) {
  return __builtin_amdgcn_wave_reduce_fmax_f32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double
__clc_sub_group_non_uniform_reduce_max(double x) {
  return __builtin_amdgcn_wave_reduce_fmax_f64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half
__clc_sub_group_non_uniform_reduce_add(half x) {
  // FIXME: There should be a direct half builtin available.
  return (float)__clc_sub_group_non_uniform_reduce_add((float)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half
__clc_sub_group_non_uniform_reduce_min(half x) {
  // FIXME: There should be a direct half builtin available.
  return (float)__clc_sub_group_non_uniform_reduce_min((float)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half
__clc_sub_group_non_uniform_reduce_max(half x) {
  // FIXME: There should be a direct half builtin available.
  return (float)__clc_sub_group_non_uniform_reduce_max((float)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_add(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_add((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_add(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_add((int)x);
}

// FIXME: There should be a direct short builtin available.
_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_add(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_add((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_add(short x) {
  return (int)__clc_sub_group_non_uniform_reduce_add((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_min(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_min((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_min(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_min((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_min(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_min((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_min(short x) {
  return (int)__clc_sub_group_non_uniform_reduce_min((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_max(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_max((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_max(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_max((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_max(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_max((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_max(short x) {
  return (int)__clc_sub_group_non_uniform_reduce_max((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_and(uint x) {
  return __builtin_amdgcn_wave_reduce_and_b32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_and(int x) {
  return (int)__clc_sub_group_non_uniform_reduce_and((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_and(ulong x) {
  return __builtin_amdgcn_wave_reduce_and_b64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_and(long x) {
  return (long)__clc_sub_group_non_uniform_reduce_and((ulong)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_or(uint x) {
  return __builtin_amdgcn_wave_reduce_or_b32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_or(int x) {
  return (int)__clc_sub_group_non_uniform_reduce_or((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_or(ulong x) {
  return __builtin_amdgcn_wave_reduce_or_b64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_or(long x) {
  return (long)__clc_sub_group_non_uniform_reduce_or((ulong)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_xor(uint x) {
  return __builtin_amdgcn_wave_reduce_xor_b32(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_xor(int x) {
  return (int)__clc_sub_group_non_uniform_reduce_xor((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_xor(ulong x) {
  return __builtin_amdgcn_wave_reduce_xor_b64(x, 0);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_xor(long x) {
  return (long)__clc_sub_group_non_uniform_reduce_xor((ulong)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_and(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_and((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_and(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_and((int)x);
}

// FIXME: There should be a direct short builtin available.
_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_and(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_and((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_and(short x) {
  return (int)__clc_sub_group_non_uniform_reduce_and((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_or(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_or((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_or(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_or((int)x);
}

// FIXME: There should be a direct short builtin available.
_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_or(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_or((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_or(short x) {
  return (int)__clc_sub_group_non_uniform_reduce_or((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_xor(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_xor((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_xor(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_xor((int)x);
}

// FIXME: There should be a direct short builtin available.
_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_xor(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_xor((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_xor(short x) {
  return (int)__clc_sub_group_non_uniform_reduce_xor((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uint
__clc_sub_group_non_uniform_reduce_mul(uint x) {
  (void)x;
  // TODO:
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_mul(int x) {
  return (int)__clc_sub_group_non_uniform_reduce_mul((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ulong
__clc_sub_group_non_uniform_reduce_mul(ulong x) {
  (void)x;
  // TODO:
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST long
__clc_sub_group_non_uniform_reduce_mul(long x) {
  return (long)__clc_sub_group_non_uniform_reduce_mul((ulong)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST char
__clc_sub_group_non_uniform_reduce_mul(char x) {
  return (char)__clc_sub_group_non_uniform_reduce_mul((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST uchar
__clc_sub_group_non_uniform_reduce_mul(uchar x) {
  return (uchar)__clc_sub_group_non_uniform_reduce_mul((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST short
__clc_sub_group_non_uniform_reduce_mul(short x) {
  return (short)__clc_sub_group_non_uniform_reduce_mul((int)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST ushort
__clc_sub_group_non_uniform_reduce_mul(ushort x) {
  return (ushort)__clc_sub_group_non_uniform_reduce_mul((uint)x);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_logical_and(int predicate) {
  // TODO:
  (void)predicate;
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_logical_or(int predicate) {
  // TODO:
  (void)predicate;
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST int
__clc_sub_group_non_uniform_reduce_logical_xor(int predicate) {
  // TODO:
  (void)predicate;
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST float
__clc_sub_group_non_uniform_reduce_mul(float x) {
  (void)x;
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST double
__clc_sub_group_non_uniform_reduce_mul(double x) {
  (void)x;
  __builtin_trap();
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST half
__clc_sub_group_non_uniform_reduce_mul(half x) {
  (void)x;
  __builtin_trap();
}
