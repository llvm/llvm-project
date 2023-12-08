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

VSTORE_ADDR_SPACES(char)
VSTORE_ADDR_SPACES(uchar)
VSTORE_ADDR_SPACES(short)
VSTORE_ADDR_SPACES(ushort)
VSTORE_ADDR_SPACES(int)
VSTORE_ADDR_SPACES(uint)
VSTORE_ADDR_SPACES(long)
VSTORE_ADDR_SPACES(ulong)
VSTORE_ADDR_SPACES(float)


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
_CLC_DEF void __clc_vstore_half_##STYPE##_helper##AS(STYPE s, AS half *d) \
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

#define VEC_STORE1(STYPE, AS, val, ROUNDF) __clc_vstore_half_##STYPE##_helper##AS (ROUNDF(val), &mem[offset++]);

#define VEC_STORE2(STYPE, AS, val, ROUNDF) \
	VEC_STORE1(STYPE, AS, val.lo, ROUNDF) \
	VEC_STORE1(STYPE, AS, val.hi, ROUNDF)
#define VEC_STORE3(STYPE, AS, val, ROUNDF) \
	VEC_STORE1(STYPE, AS, val.s0, ROUNDF) \
	VEC_STORE1(STYPE, AS, val.s1, ROUNDF) \
	VEC_STORE1(STYPE, AS, val.s2, ROUNDF)
#define VEC_STORE4(STYPE, AS, val, ROUNDF) \
	VEC_STORE2(STYPE, AS, val.lo, ROUNDF) \
	VEC_STORE2(STYPE, AS, val.hi, ROUNDF)
#define VEC_STORE8(STYPE, AS, val, ROUNDF) \
	VEC_STORE4(STYPE, AS, val.lo, ROUNDF) \
	VEC_STORE4(STYPE, AS, val.hi, ROUNDF)
#define VEC_STORE16(STYPE, AS, val, ROUNDF) \
	VEC_STORE8(STYPE, AS, val.lo, ROUNDF) \
	VEC_STORE8(STYPE, AS, val.hi, ROUNDF)

#define __FUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, STYPE, AS, ROUNDF) \
  _CLC_OVERLOAD _CLC_DEF void vstore_half##SUFFIX(TYPE vec, size_t offset, AS half *mem) { \
    offset *= VEC_SIZE; \
    VEC_STORE##VEC_SIZE(STYPE, AS, vec, ROUNDF) \
  } \
  _CLC_OVERLOAD _CLC_DEF void vstorea_half##SUFFIX(TYPE vec, size_t offset, AS half *mem) { \
    offset *= OFFSET; \
    VEC_STORE##VEC_SIZE(STYPE, AS, vec, ROUNDF) \
  }

_CLC_DEF _CLC_OVERLOAD float __clc_noop(float x)
{
	return x;
}
_CLC_DEF _CLC_OVERLOAD float __clc_rtz(float x)
{
	/* Remove lower 13 bits to make sure the number is rounded down */
	int mask = 0xffffe000;
	const int exp = (as_uint(x) >> 23 & 0xff) - 127;
	/* Denormals cannot be flushed, and they use different bit for rounding */
	if (exp < -14)
		mask <<= min(-(exp + 14), 10);
	/* RTZ does not produce Inf for large numbers */
	if (fabs(x) > 65504.0f && !isinf(x))
		return copysign(65504.0f, x);
	/* Handle nan corner case */
	if (isnan(x))
		return x;
	return as_float(as_uint(x) & mask);
}
_CLC_DEF _CLC_OVERLOAD float __clc_rti(float x)
{
	const float inf = copysign(INFINITY, x);
	/* Set lower 13 bits */
	int mask = (1 << 13) - 1;
	const int exp = (as_uint(x) >> 23 & 0xff) - 127;
	/* Denormals cannot be flushed, and they use different bit for rounding */
	if (exp < -14)
		mask = (1 << (13 + min(-(exp + 14), 10))) - 1;
	/* Handle nan corner case */
	if (isnan(x))
		return x;
	const float next = nextafter(as_float(as_uint(x) | mask), inf);
	return ((as_uint(x) & mask) == 0) ? x : next;
}
_CLC_DEF _CLC_OVERLOAD float __clc_rtn(float x)
{
	return ((as_uint(x) & 0x80000000) == 0) ? __clc_rtz(x) : __clc_rti(x);
}
_CLC_DEF _CLC_OVERLOAD float __clc_rtp(float x)
{
	return ((as_uint(x) & 0x80000000) == 0) ? __clc_rti(x) : __clc_rtz(x);
}
_CLC_DEF _CLC_OVERLOAD float __clc_rte(float x)
{
	/* Mantisa + implicit bit */
	const uint mantissa = (as_uint(x) & 0x7fffff) | (1u << 23);
	const int exp = (as_uint(x) >> 23 & 0xff) - 127;
	int shift = 13;
	if (exp < -14) {
		/* The default assumes lower 13 bits are rounded,
		 * but it might be more for denormals.
		 * Shifting beyond last == 0b, and qr == 00b is not necessary */
		shift += min(-(exp + 14), 15);
	}
	int mask = (1 << shift) - 1;
	const uint grs = mantissa & mask;
	const uint last = mantissa & (1 << shift);
	/* IEEE round up rule is: grs > 101b or grs == 100b and last == 1.
	 * exp > 15 should round to inf. */
	bool roundup = (grs > (1 << (shift - 1))) ||
		(grs == (1 << (shift - 1)) && last != 0) || (exp > 15);
	return roundup ? __clc_rti(x) : __clc_rtz(x);
}

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_noop(double x)
{
	return x;
}
_CLC_DEF _CLC_OVERLOAD double __clc_rtz(double x)
{
	/* Remove lower 42 bits to make sure the number is rounded down */
	ulong mask = 0xfffffc0000000000UL;
	const int exp = (as_ulong(x) >> 52 & 0x7ff) - 1023;
	/* Denormals cannot be flushed, and they use different bit for rounding */
	if (exp < -14)
		mask <<= min(-(exp + 14), 10);
	/* RTZ does not produce Inf for large numbers */
	if (fabs(x) > 65504.0 && !isinf(x))
		return copysign(65504.0, x);
	/* Handle nan corner case */
	if (isnan(x))
		return x;
	return as_double(as_ulong(x) & mask);
}
_CLC_DEF _CLC_OVERLOAD double __clc_rti(double x)
{
	const double inf = copysign((double)INFINITY, x);
	/* Set lower 42 bits */
	long mask = (1UL << 42UL) - 1UL;
	const int exp = (as_ulong(x) >> 52 & 0x7ff) - 1023;
	/* Denormals cannot be flushed, and they use different bit for rounding */
	if (exp < -14)
		mask = (1UL << (42UL + min(-(exp + 14), 10))) - 1;
	/* Handle nan corner case */
	if (isnan(x))
		return x;
	const double next = nextafter(as_double(as_ulong(x) | mask), inf);
	return ((as_ulong(x) & mask) == 0) ? x : next;
}
_CLC_DEF _CLC_OVERLOAD double __clc_rtn(double x)
{
	return ((as_ulong(x) & 0x8000000000000000UL) == 0) ? __clc_rtz(x) : __clc_rti(x);
}
_CLC_DEF _CLC_OVERLOAD double __clc_rtp(double x)
{
	return ((as_ulong(x) & 0x8000000000000000UL) == 0) ? __clc_rti(x) : __clc_rtz(x);
}
_CLC_DEF _CLC_OVERLOAD double __clc_rte(double x)
{
	/* Mantisa + implicit bit */
	const ulong mantissa = (as_ulong(x) & 0xfffffffffffff) | (1UL << 52);
	const int exp = (as_ulong(x) >> 52 & 0x7ff) - 1023;
	int shift = 42;
	if (exp < -14) {
		/* The default assumes lower 13 bits are rounded,
		 * but it might be more for denormals.
		 * Shifting beyond last == 0b, and qr == 00b is not necessary */
		shift += min(-(exp + 14), 15);
	}
	ulong mask = (1UL << shift) - 1UL;
	const ulong grs = mantissa & mask;
	const ulong last = mantissa & (1UL << shift);
	/* IEEE round up rule is: grs > 101b or grs == 100b and last == 1.
	 * exp > 15 should round to inf. */
	bool roundup = (grs > (1UL << (shift - 1UL))) ||
		(grs == (1UL << (shift - 1UL)) && last != 0) || (exp > 15);
	return roundup ? __clc_rti(x) : __clc_rtz(x);
}
#endif

#define __XFUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, STYPE, AS) \
	__FUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, STYPE, AS, __clc_noop) \
	__FUNC(SUFFIX ## _rtz, VEC_SIZE, OFFSET, TYPE, STYPE, AS, __clc_rtz) \
	__FUNC(SUFFIX ## _rtn, VEC_SIZE, OFFSET, TYPE, STYPE, AS, __clc_rtn) \
	__FUNC(SUFFIX ## _rtp, VEC_SIZE, OFFSET, TYPE, STYPE, AS, __clc_rtp) \
	__FUNC(SUFFIX ## _rte, VEC_SIZE, OFFSET, TYPE, STYPE, AS, __clc_rte)

#define FUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, STYPE, AS) \
	__XFUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, STYPE, AS)

#define __CLC_BODY "vstore_half.inc"
#include <clc/math/gentype.inc>
#undef __CLC_BODY
#undef FUNC
#undef __XFUNC
#undef __FUNC
#undef VEC_LOAD16
#undef VEC_LOAD8
#undef VEC_LOAD4
#undef VEC_LOAD3
#undef VEC_LOAD2
#undef VEC_LOAD1
#undef DECLARE_HELPER
#undef VSTORE_ADDR_SPACES
#undef VSTORE_VECTORIZE
