#include <clc/clc.h>

#define VLOAD_VECTORIZE(PRIM_TYPE, ADDR_SPACE) \
  typedef PRIM_TYPE##2 less_aligned_##ADDR_SPACE##PRIM_TYPE##2 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##2 vload2(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2*) (&x[2*offset])); \
  } \
\
  typedef PRIM_TYPE##3 less_aligned_##ADDR_SPACE##PRIM_TYPE##3 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##3 vload3(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    PRIM_TYPE##2 vec = *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2*) (&x[3*offset])); \
    return (PRIM_TYPE##3)(vec.s0, vec.s1, x[offset*3+2]); \
  } \
\
  typedef PRIM_TYPE##4 less_aligned_##ADDR_SPACE##PRIM_TYPE##4 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##4 vload4(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##4*) (&x[4*offset])); \
  } \
\
  typedef PRIM_TYPE##8 less_aligned_##ADDR_SPACE##PRIM_TYPE##8 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##8 vload8(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##8*) (&x[8*offset])); \
  } \
\
  typedef PRIM_TYPE##16 less_aligned_##ADDR_SPACE##PRIM_TYPE##16 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF PRIM_TYPE##16 vload16(size_t offset, const ADDR_SPACE PRIM_TYPE *x) { \
    return *((const ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##16*) (&x[16*offset])); \
  } \

#define VLOAD_ADDR_SPACES(__CLC_SCALAR_GENTYPE) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __private) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __local) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __constant) \
    VLOAD_VECTORIZE(__CLC_SCALAR_GENTYPE, __global) \

#define VLOAD_TYPES() \
    VLOAD_ADDR_SPACES(char) \
    VLOAD_ADDR_SPACES(uchar) \
    VLOAD_ADDR_SPACES(short) \
    VLOAD_ADDR_SPACES(ushort) \
    VLOAD_ADDR_SPACES(int) \
    VLOAD_ADDR_SPACES(uint) \
    VLOAD_ADDR_SPACES(long) \
    VLOAD_ADDR_SPACES(ulong) \
    VLOAD_ADDR_SPACES(float) \

VLOAD_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    VLOAD_ADDR_SPACES(double)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
    VLOAD_ADDR_SPACES(half)
#endif

/* vload_half are legal even without cl_khr_fp16 */
/* no vload_half for double */
#if __clang_major__ < 6
float __clc_vload_half_float_helper__constant(const __constant half *);
float __clc_vload_half_float_helper__global(const __global half *);
float __clc_vload_half_float_helper__local(const __local half *);
float __clc_vload_half_float_helper__private(const __private half *);

#define VEC_LOAD1(val, AS) val = __clc_vload_half_float_helper##AS (&mem[offset++]);
#else
#define VEC_LOAD1(val, AS) val = __builtin_load_halff(&mem[offset++]);
#endif

#define VEC_LOAD2(val, AS) \
	VEC_LOAD1(val.lo, AS) \
	VEC_LOAD1(val.hi, AS)
#define VEC_LOAD3(val, AS) \
	VEC_LOAD1(val.s0, AS) \
	VEC_LOAD1(val.s1, AS) \
	VEC_LOAD1(val.s2, AS)
#define VEC_LOAD4(val, AS) \
	VEC_LOAD2(val.lo, AS) \
	VEC_LOAD2(val.hi, AS)
#define VEC_LOAD8(val, AS) \
	VEC_LOAD4(val.lo, AS) \
	VEC_LOAD4(val.hi, AS)
#define VEC_LOAD16(val, AS) \
	VEC_LOAD8(val.lo, AS) \
	VEC_LOAD8(val.hi, AS)

#define __FUNC(SUFFIX, VEC_SIZE, OFFSET_SIZE, TYPE, AS) \
  _CLC_OVERLOAD _CLC_DEF TYPE vload_half##SUFFIX(size_t offset, const AS half *mem) { \
    offset *= VEC_SIZE; \
    TYPE __tmp; \
    VEC_LOAD##VEC_SIZE(__tmp, AS) \
    return __tmp; \
  } \
  _CLC_OVERLOAD _CLC_DEF TYPE vloada_half##SUFFIX(size_t offset, const AS half *mem) { \
    offset *= OFFSET_SIZE; \
    TYPE __tmp; \
    VEC_LOAD##VEC_SIZE(__tmp, AS) \
    return __tmp; \
  }

#define FUNC(SUFFIX, VEC_SIZE, OFFSET_SIZE, TYPE, AS) __FUNC(SUFFIX, VEC_SIZE, OFFSET_SIZE, TYPE, AS)

#define __CLC_BODY "vload_half.inc"
#include <clc/math/gentype.inc>
#undef __CLC_BODY
#undef FUNC
#undef __FUNC
#undef VEC_LOAD16
#undef VEC_LOAD8
#undef VEC_LOAD4
#undef VEC_LOAD3
#undef VEC_LOAD2
#undef VEC_LOAD1
#undef VLOAD_TYPES
#undef VLOAD_ADDR_SPACES
#undef VLOAD_VECTORIZE
