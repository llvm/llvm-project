/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define ATTR __attribute__((overloadable))

// Cast away volatile before calling clang builtin
#define VOLATILE

#define AC_int(X) X
#define AC_uint(X) X
#define AC_long(X) X
#define AC_ulong(X) X
#define AC_intptr_t(X) X
#define AC_uintptr_t(X) X
#define AC_size_t(X) X
#define AC_ptrdiff_t(X) X
#define AC_float(X) as_int(X)
#define AC_double(X) as_long(X)

#define RC_int(X) X
#define RC_uint(X) X
#define RC_long(X) X
#define RC_ulong(X) X
#define RC_intptr_t(X) X
#define RC_uintptr_t(X) X
#define RC_size_t(X) X
#define RC_ptrdiff_t(X) X
#define RC_float(X) as_float(X)
#define RC_double(X) as_double(X)

#define PC_int (VOLATILE atomic_int *)
#define PC_uint (VOLATILE atomic_uint *)
#define PC_long (VOLATILE atomic_long *)
#define PC_ulong (VOLATILE atomic_ulong *)
#define PC_intptr_t (VOLATILE atomic_intptr_t *)
#define PC_uintptr_t (VOLATILE atomic_uintptr_t *)
#define PC_size_t (VOLATILE atomic_size_t *)
#define PC_ptrdiff_t (VOLATILE atomic_ptrdiff_t *)
#define PC_float (VOLATILE atomic_int *)
#define PC_double (VOLATILE atomic_long *)

#define EC_int
#define EC_uint
#define EC_long
#define EC_ulong
#define EC_intptr_t
#define EC_uintptr_t
#define EC_size_t
#define EC_ptrdiff_t
#define EC_float (int *)
#define EC_double (long *)

#define OCL12_MEMORY_ORDER memory_order_relaxed
#define OCL12_MEMORY_SCOPE memory_scope_device

#define F_inc __opencl_atomic_fetch_add
#define F_dec __opencl_atomic_fetch_sub

// extension and 1.2 functions
#define GEN1(T,A,O) \
ATTR T \
atom_##O(volatile A T *p, T v) \
{ \
    return __opencl_atomic_fetch_##O((VOLATILE atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define GEN2(T,A,O) \
ATTR T \
atomic_##O(volatile A T *p, T v) \
{ \
    return __opencl_atomic_fetch_##O((VOLATILE atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define OPSA(F,T,A) \
    F(T,A,add) \
    F(T,A,sub) \
    F(T,A,max) \
    F(T,A,min) \
    F(T,A,and) \
    F(T,A,or) \
    F(T,A,xor)

#define OPS(F,T) \
    OPSA(F,T,__local) \
    OPSA(F,T,__global) \
    OPSA(F,T,)

#define ALL() \
    OPS(GEN1,int) \
    OPS(GEN2,int) \
    OPS(GEN1,uint) \
    OPS(GEN2,uint) \
    OPS(GEN1,long) \
    OPS(GEN1,ulong)

ALL()

// Handle inc and dec
#undef GEN1
#undef GEN2
#undef OPSA

#define OPSA(F,T,A) \
    F(T,A,inc) \
    F(T,A,dec)


#define GEN1(T,A,O) \
ATTR T \
atom_##O(volatile A T *p) \
{ \
    return F_##O((VOLATILE atomic_##T *)p, (T)1, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define GEN2(T,A,O) \
ATTR T \
atomic_##O(volatile A T *p) \
{ \
    return F_##O((VOLATILE atomic_##T *)p, (T)1, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

ALL()

// Handle xchg
#undef GEN1
#undef GEN2
#undef OPSA
#undef OPS

#define GEN1(T,A) \
ATTR T \
atom_xchg(volatile A T *p, T v) \
{ \
    return __opencl_atomic_exchange((VOLATILE atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define GEN2(T,A) \
ATTR T \
atomic_xchg(volatile A T *p, T v) \
{ \
    return __opencl_atomic_exchange((VOLATILE atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define OPS(F,T) \
    F(T,__local) \
    F(T,__global) \
    F(T,) \

ALL()

#define G(A) \
ATTR float \
atomic_xchg(volatile A float *p, float v) \
{ \
    return as_float(__opencl_atomic_exchange((VOLATILE atomic_int *)p, as_int(v), OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE)); \
}

G(__local)
G(__global)
G()

// Handle cmpxchg
#undef GEN1
#undef GEN2
#undef G

#define GEN1(T,A) \
ATTR T \
atom_cmpxchg(volatile A T *p, T e, T d) \
{ \
    __opencl_atomic_compare_exchange_strong((VOLATILE atomic_##T *)p, &e, d,  OCL12_MEMORY_ORDER, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
    return e; \
}

#define GEN2(T,A) \
ATTR T \
atomic_cmpxchg(volatile A T *p, T e, T d) \
{ \
    __opencl_atomic_compare_exchange_strong((VOLATILE atomic_##T *)p, &e, d,  OCL12_MEMORY_ORDER, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
    return e; \
}

ALL()
#undef GEN1
#undef GEN2
#undef ALL

// 2.0 functions

#define GENI(T) \
ATTR void \
atomic_init(volatile atomic_##T *p, T v) \
{ \
    __opencl_atomic_init((VOLATILE atomic_##T *)p, v); \
}

#define GENS(T) \
ATTR void \
atomic_store(volatile atomic_##T *p, T v) \
{ \
    __opencl_atomic_store((VOLATILE atomic_##T *)p, v, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR void \
atomic_store_explicit(volatile atomic_##T *p, T v, memory_order o) \
{ \
    __opencl_atomic_store((VOLATILE atomic_##T *)p, v, o, memory_scope_device); \
} \
 \
ATTR void \
atomic_store_explicit(volatile atomic_##T *p, T v, memory_order o, memory_scope s) \
{ \
    __opencl_atomic_store((VOLATILE atomic_##T *)p, v, o, s); \
}

#define GENL(T) \
ATTR T \
atomic_load(volatile atomic_##T *p) \
{ \
    return __opencl_atomic_load((VOLATILE atomic_##T *)p, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR T \
atomic_load_explicit(volatile atomic_##T *p, memory_order o) \
{ \
    return __opencl_atomic_load((VOLATILE atomic_##T *)p, o, memory_scope_device); \
} \
 \
ATTR T \
atomic_load_explicit(volatile atomic_##T *p, memory_order o, memory_scope s) \
{ \
    return __opencl_atomic_load((VOLATILE atomic_##T *)p, o, s); \
}

#define GENX(T) \
ATTR T \
atomic_exchange(volatile atomic_##T *p, T v) \
{ \
    return RC_##T(__opencl_atomic_exchange(PC_##T p, AC_##T(v), memory_order_seq_cst, memory_scope_device)); \
} \
 \
ATTR T \
atomic_exchange_explicit(volatile atomic_##T *p, T v, memory_order o) \
{ \
    return RC_##T(__opencl_atomic_exchange(PC_##T p, AC_##T(v), o, memory_scope_device)); \
} \
 \
ATTR T \
atomic_exchange_explicit(volatile atomic_##T *p, T v, memory_order o, memory_scope s) \
{ \
    return RC_##T(__opencl_atomic_exchange(PC_##T p, AC_##T(v), o, s)); \
}

#define GENCX(T,K) \
ATTR bool \
atomic_compare_exchange_##K(volatile atomic_##T *p, T *e, T d) \
{ \
    return __opencl_atomic_compare_exchange_##K(PC_##T p, EC_##T e, AC_##T(d), memory_order_seq_cst, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR bool \
atomic_compare_exchange_##K##_explicit(volatile atomic_##T *p, T *e, T d, memory_order os, memory_order of) \
{ \
    return __opencl_atomic_compare_exchange_##K(PC_##T p, EC_##T e, AC_##T(d), os, of, memory_scope_device); \
} \
 \
ATTR bool \
atomic_compare_exchange_##K##_explicit(volatile atomic_##T *p, T *e, T d, memory_order os, memory_order of, memory_scope s) \
{ \
    return __opencl_atomic_compare_exchange_##K(PC_##T p, EC_##T e, AC_##T(d), os, of, s); \
}

#define GENFO(T,O) \
ATTR T \
atomic_fetch_##O(volatile atomic_##T *p, T v) \
{ \
    return RC_##T(__opencl_atomic_fetch_##O(PC_##T p, AC_##T(v), memory_order_seq_cst, memory_scope_device)); \
} \
 \
ATTR T \
atomic_fetch_##O##_explicit(volatile atomic_##T *p, T v, memory_order o) \
{ \
    return RC_##T(__opencl_atomic_fetch_##O(PC_##T p, AC_##T(v), o, memory_scope_device)); \
} \
 \
ATTR T \
atomic_fetch_##O##_explicit(volatile atomic_##T *p, T v, memory_order o, memory_scope s) \
{ \
    return RC_##T(__opencl_atomic_fetch_##O(PC_##T p, AC_##T(v), o, s)); \
}

#define CX(T) \
    GENCX(T,strong) \
    GENCX(T,weak)

#define FO(T) \
    GENFO(T,add) \
    GENFO(T,sub) \
    GENFO(T,or) \
    GENFO(T,xor) \
    GENFO(T,and) \
    GENFO(T,min) \
    GENFO(T,max) \

#define ALLI(F) \
    F(int) \
    F(uint) \
    F(long) \
    F(ulong)

#define ALL(F) \
    ALLI(F) \
    F(float) \
    F(double)

ALL(GENI)
ALL(GENL)
ALL(GENS)
ALL(GENX)
ALL(CX)
ALLI(FO)

// These are needed for uintptr_t
ATTR ulong
atomic_fetch_add(volatile atomic_ulong *p, long v)
{
    return __opencl_atomic_fetch_add((VOLATILE atomic_ulong *)p, (ulong)v, memory_order_seq_cst, memory_scope_device);
}

ATTR ulong
atomic_fetch_add_explicit(volatile atomic_ulong *p, long v, memory_order o)
{
    return __opencl_atomic_fetch_add((VOLATILE atomic_ulong *)p, (ulong)v, o, memory_scope_device);
}

ATTR ulong
atomic_fetch_add_explicit(volatile atomic_ulong *p, long v, memory_order o, memory_scope s)
{
    return __opencl_atomic_fetch_add((VOLATILE atomic_ulong *)p, (ulong)v, o, s);
}

ATTR ulong
atomic_fetch_sub(volatile atomic_ulong *p, long v)
{
    return __opencl_atomic_fetch_sub((VOLATILE atomic_ulong *)p, (ulong)v, memory_order_seq_cst, memory_scope_device);
}

ATTR ulong
atomic_fetch_sub_explicit(volatile atomic_ulong *p, long v, memory_order o)
{
    return __opencl_atomic_fetch_sub((VOLATILE atomic_ulong *)p, (ulong)v, o, memory_scope_device);
}

ATTR ulong
atomic_fetch_sub_explicit(volatile atomic_ulong *p, long v, memory_order o, memory_scope s)
{
    return __opencl_atomic_fetch_sub((VOLATILE atomic_ulong *)p, (ulong)v, o, s);
}

// flag functions
ATTR bool
atomic_flag_test_and_set(volatile atomic_flag *p)
{
    return __opencl_atomic_exchange((VOLATILE atomic_int *)p, 1, memory_order_seq_cst, memory_scope_device);
}

ATTR bool
atomic_flag_test_and_set_explicit(volatile atomic_flag *p, memory_order o)
{
    return __opencl_atomic_exchange((VOLATILE atomic_int *)p, 1, o, memory_scope_device);
}

ATTR bool
atomic_flag_test_and_set_explicit(volatile atomic_flag *p, memory_order o, memory_scope s)
{
    return __opencl_atomic_exchange((VOLATILE atomic_int *)p, 1, o, s);
}

ATTR void
atomic_flag_clear(volatile atomic_flag *p)
{
    __opencl_atomic_store((VOLATILE atomic_int *)p, 0, memory_order_seq_cst, memory_scope_device);
}

ATTR void
atomic_flag_clear_explicit(volatile atomic_flag *p, memory_order o)
{
    __opencl_atomic_store((VOLATILE atomic_int *)p, 0, o, memory_scope_device);
}

ATTR void
atomic_flag_clear_explicit(volatile atomic_flag *p, memory_order o, memory_scope s)
{
    __opencl_atomic_store((VOLATILE atomic_int *)p, 0, o, s);
}

