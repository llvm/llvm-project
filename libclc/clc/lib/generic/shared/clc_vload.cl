//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/shared/clc_vload.h>

#define __CLC_VLOAD_VECTORIZE(PRIM_TYPE, ADDR_SPACE)                           \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##2 __clc_vload2(                            \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *(                                                                  \
        (const ADDR_SPACE less_aligned_##PRIM_TYPE##2 *)(&x[2 * offset]));     \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##3 __clc_vload3(                            \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    PRIM_TYPE##2 vec =                                                         \
        *((const ADDR_SPACE less_aligned_##PRIM_TYPE##2 *)(&x[3 * offset]));   \
    return (PRIM_TYPE##3)(vec.s0, vec.s1, x[offset * 3 + 2]);                  \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##4 __clc_vload4(                            \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *(                                                                  \
        (const ADDR_SPACE less_aligned_##PRIM_TYPE##4 *)(&x[4 * offset]));     \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##8 __clc_vload8(                            \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *(                                                                  \
        (const ADDR_SPACE less_aligned_##PRIM_TYPE##8 *)(&x[8 * offset]));     \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##16 __clc_vload16(                          \
      size_t offset, const ADDR_SPACE PRIM_TYPE *x) {                          \
    return *(                                                                  \
        (const ADDR_SPACE less_aligned_##PRIM_TYPE##16 *)(&x[16 * offset]));   \
  }

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_VLOAD_VECTORIZE_GENERIC __CLC_VLOAD_VECTORIZE
#else
// The generic address space isn't available, so make the macro do nothing
#define __CLC_VLOAD_VECTORIZE_GENERIC(X, Y)
#endif

#define __CLC_VLOAD_ADDR_SPACES(__CLC_SCALAR_GENTYPE)                          \
  __CLC_VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __private)                       \
  __CLC_VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __local)                         \
  __CLC_VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __constant)                      \
  __CLC_VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __global)                        \
  __CLC_VLOAD_VECTORIZE_GENERIC(__CLC_SCALAR_GENTYPE, __generic)

#define __CLC_VLOAD_TYPES()                                                    \
  __CLC_VLOAD_ADDR_SPACES(char)                                                \
  __CLC_VLOAD_ADDR_SPACES(uchar)                                               \
  __CLC_VLOAD_ADDR_SPACES(short)                                               \
  __CLC_VLOAD_ADDR_SPACES(ushort)                                              \
  __CLC_VLOAD_ADDR_SPACES(int)                                                 \
  __CLC_VLOAD_ADDR_SPACES(uint)                                                \
  __CLC_VLOAD_ADDR_SPACES(long)                                                \
  __CLC_VLOAD_ADDR_SPACES(ulong)                                               \
  __CLC_VLOAD_ADDR_SPACES(float)

__CLC_VLOAD_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__CLC_VLOAD_ADDR_SPACES(double)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__CLC_VLOAD_ADDR_SPACES(half)
#endif

/* vload_half are legal even without cl_khr_fp16 */
/* no vload_half for double */
#define __CLC_VEC_LOAD1(val, AS) val = __builtin_load_halff(&mem[offset++]);
#define __CLC_VEC_LOAD2(val, AS)                                               \
  __CLC_VEC_LOAD1(val.lo, AS)                                                  \
  __CLC_VEC_LOAD1(val.hi, AS)
#define __CLC_VEC_LOAD3(val, AS)                                               \
  __CLC_VEC_LOAD1(val.s0, AS)                                                  \
  __CLC_VEC_LOAD1(val.s1, AS)                                                  \
  __CLC_VEC_LOAD1(val.s2, AS)
#define __CLC_VEC_LOAD4(val, AS)                                               \
  __CLC_VEC_LOAD2(val.lo, AS)                                                  \
  __CLC_VEC_LOAD2(val.hi, AS)
#define __CLC_VEC_LOAD8(val, AS)                                               \
  __CLC_VEC_LOAD4(val.lo, AS)                                                  \
  __CLC_VEC_LOAD4(val.hi, AS)
#define __CLC_VEC_LOAD16(val, AS)                                              \
  __CLC_VEC_LOAD8(val.lo, AS)                                                  \
  __CLC_VEC_LOAD8(val.hi, AS)

#define __CLC_FUNC_IMPL(SUFFIX, VEC_SIZE, OFFSET_SIZE, TYPE, AS)               \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_vload_half##SUFFIX(size_t offset,          \
                                                       const AS half *mem) {   \
    offset *= VEC_SIZE;                                                        \
    TYPE __tmp;                                                                \
    __CLC_VEC_LOAD##VEC_SIZE(__tmp, AS) return __tmp;                          \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF TYPE __clc_vloada_half##SUFFIX(size_t offset,         \
                                                        const AS half *mem) {  \
    offset *= OFFSET_SIZE;                                                     \
    TYPE __tmp;                                                                \
    __CLC_VEC_LOAD##VEC_SIZE(__tmp, AS) return __tmp;                          \
  }

#define __CLC_FUNC(SUFFIX, VEC_SIZE, OFFSET_SIZE, TYPE, AS)                    \
  __CLC_FUNC_IMPL(SUFFIX, VEC_SIZE, OFFSET_SIZE, TYPE, AS)

#define __CLC_BODY "clc_vload_half.inc"
#include <clc/math/gentype.inc>
#undef __CLC_FUNC
#undef __CLC_FUNC_IMPL
#undef __CLC_VEC_LOAD16
#undef __CLC_VEC_LOAD8
#undef __CLC_VEC_LOAD4
#undef __CLC_VEC_LOAD3
#undef __CLC_VEC_LOAD2
#undef __CLC_VEC_LOAD1
#undef __CLC_VLOAD_TYPES
#undef __CLC_VLOAD_ADDR_SPACES
#undef __CLC_VLOAD_VECTORIZE
#undef __CLC_VLOAD_VECTORIZE_GENERIC
