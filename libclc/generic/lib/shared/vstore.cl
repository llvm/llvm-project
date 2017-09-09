#include <clc/clc.h>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define VSTORE_VECTORIZE(PRIM_TYPE, ADDR_SPACE) \
  typedef PRIM_TYPE##2 less_aligned_##ADDR_SPACE##PRIM_TYPE##2 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF void vstore2(PRIM_TYPE##2 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2*) (&mem[2*offset])) = vec; \
  } \
\
  _CLC_OVERLOAD _CLC_DEF void vstore3(PRIM_TYPE##3 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2*) (&mem[3*offset])) = (PRIM_TYPE##2)(vec.s0, vec.s1); \
    mem[3 * offset + 2] = vec.s2;\
  } \
\
  typedef PRIM_TYPE##4 less_aligned_##ADDR_SPACE##PRIM_TYPE##4 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF void vstore4(PRIM_TYPE##4 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##4*) (&mem[4*offset])) = vec; \
  } \
\
  typedef PRIM_TYPE##8 less_aligned_##ADDR_SPACE##PRIM_TYPE##8 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF void vstore8(PRIM_TYPE##8 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##8*) (&mem[8*offset])) = vec; \
  } \
\
  typedef PRIM_TYPE##16 less_aligned_##ADDR_SPACE##PRIM_TYPE##16 __attribute__ ((aligned (sizeof(PRIM_TYPE))));\
  _CLC_OVERLOAD _CLC_DEF void vstore16(PRIM_TYPE##16 vec, size_t offset, ADDR_SPACE PRIM_TYPE *mem) { \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##16*) (&mem[16*offset])) = vec; \
  } \

#define VSTORE_ADDR_SPACES(__CLC_SCALAR___CLC_GENTYPE) \
    VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __private) \
    VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __local) \
    VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __global) \

#define VSTORE_TYPES() \
    VSTORE_ADDR_SPACES(char) \
    VSTORE_ADDR_SPACES(uchar) \
    VSTORE_ADDR_SPACES(short) \
    VSTORE_ADDR_SPACES(ushort) \
    VSTORE_ADDR_SPACES(int) \
    VSTORE_ADDR_SPACES(uint) \
    VSTORE_ADDR_SPACES(long) \
    VSTORE_ADDR_SPACES(ulong) \
    VSTORE_ADDR_SPACES(float) \

VSTORE_TYPES()

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
    VSTORE_ADDR_SPACES(double)
#endif
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
    VSTORE_ADDR_SPACES(half)
#endif

/* vstore_half are legal even without cl_khr_fp16 */
#if __clang_major__ < 6
#define DECLARE_HELPER(STYPE, AS, builtin) void __clc_vstore_half_##STYPE##_helper##AS(STYPE, AS half *);
#else
#define DECLARE_HELPER(STYPE, AS, __builtin) \
inline void __clc_vstore_half_##STYPE##_helper##AS(STYPE s, AS half *d) \
{ \
	__builtin(s, d); \
}
#endif

DECLARE_HELPER(float, __private, __builtin_store_halff);
DECLARE_HELPER(float, __global, __builtin_store_halff);
DECLARE_HELPER(float, __local, __builtin_store_halff);

#ifdef cl_khr_fp64
DECLARE_HELPER(double, __private, __builtin_store_half);
DECLARE_HELPER(double, __global, __builtin_store_half);
DECLARE_HELPER(double, __local, __builtin_store_half);
#endif

#define VEC_STORE1(STYPE, AS, val) __clc_vstore_half_##STYPE##_helper##AS (val, &mem[offset++]);

#define VEC_STORE2(STYPE, AS, val) \
	VEC_STORE1(STYPE, AS, val.lo) \
	VEC_STORE1(STYPE, AS, val.hi)
#define VEC_STORE3(STYPE, AS, val) \
	VEC_STORE1(STYPE, AS, val.s0) \
	VEC_STORE1(STYPE, AS, val.s1) \
	VEC_STORE1(STYPE, AS, val.s2)
#define VEC_STORE4(STYPE, AS, val) \
	VEC_STORE2(STYPE, AS, val.lo) \
	VEC_STORE2(STYPE, AS, val.hi)
#define VEC_STORE8(STYPE, AS, val) \
	VEC_STORE4(STYPE, AS, val.lo) \
	VEC_STORE4(STYPE, AS, val.hi)
#define VEC_STORE16(STYPE, AS, val) \
	VEC_STORE8(STYPE, AS, val.lo) \
	VEC_STORE8(STYPE, AS, val.hi)

#define __FUNC(SUFFIX, VEC_SIZE, TYPE, STYPE, AS) \
  _CLC_OVERLOAD _CLC_DEF void vstore_half##SUFFIX(TYPE vec, size_t offset, AS half *mem) { \
    offset *= VEC_SIZE; \
    VEC_STORE##VEC_SIZE(STYPE, AS, vec) \
  }

#define FUNC(SUFFIX, VEC_SIZE, TYPE, STYPE, AS) __FUNC(SUFFIX, VEC_SIZE, TYPE, STYPE, AS)

#define __CLC_BODY "vstore_half.inc"
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
#undef DECLARE_HELPER
#undef VSTORE_TYPES
#undef VSTORE_ADDR_SPACES
#undef VSTORE_VECTORIZE
