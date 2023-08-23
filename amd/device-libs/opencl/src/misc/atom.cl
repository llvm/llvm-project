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

#define AT_int atomic_int
#define AT_uint atomic_uint
#define AT_long atomic_long
#define AT_ulong atomic_ulong
#define AT_intptr_t atomic_intptr_t
#define AT_uintptr_t atomic_uintptr_t
#define AT_size_t atomic_size_t
#define AT_ptrdiff_t atomic_ptrdiff_t
#define AT_float atomic_int
#define AT_double atomic_long

#define ET_int int
#define ET_uint uint
#define ET_long long
#define ET_ulong ulong
#define ET_intptr_t intptr_t
#define ET_uintptr_t uintptr_t
#define ET_size_t size_t
#define ET_ptrdiff_t ptrdiff_t
#define ET_float int
#define ET_double long

#define OCL12_MEMORY_ORDER memory_order_relaxed
#define OCL12_MEMORY_SCOPE memory_scope_device

#define F_inc __opencl_atomic_fetch_add
#define F_dec __opencl_atomic_fetch_sub

// extension and 1.2 functions
#define GEN1(T,A,O) \
ATTR T \
atom_##O(volatile A T *p, T v) \
{ \
    return __opencl_atomic_fetch_##O((VOLATILE A atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define GEN2(T,A,O) \
ATTR T \
atomic_##O(volatile A T *p, T v) \
{ \
    return __opencl_atomic_fetch_##O((VOLATILE A atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
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
    return F_##O((VOLATILE A atomic_##T *)p, (T)1, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define GEN2(T,A,O) \
ATTR T \
atomic_##O(volatile A T *p) \
{ \
    return F_##O((VOLATILE A atomic_##T *)p, (T)1, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
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
    return __opencl_atomic_exchange((VOLATILE A atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
}

#define GEN2(T,A) \
ATTR T \
atomic_xchg(volatile A T *p, T v) \
{ \
    return __opencl_atomic_exchange((VOLATILE A atomic_##T *)p, v, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
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
    return as_float(__opencl_atomic_exchange((VOLATILE A atomic_int *)p, as_int(v), OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE)); \
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
    __opencl_atomic_compare_exchange_strong((VOLATILE A atomic_##T *)p, &e, d,  OCL12_MEMORY_ORDER, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
    return e; \
}

#define GEN2(T,A) \
ATTR T \
atomic_cmpxchg(volatile A T *p, T e, T d) \
{ \
    __opencl_atomic_compare_exchange_strong((VOLATILE A atomic_##T *)p, &e, d,  OCL12_MEMORY_ORDER, OCL12_MEMORY_ORDER, OCL12_MEMORY_SCOPE); \
    return e; \
}

ALL()
#undef GEN1
#undef GEN2
#undef ALL

// 2.0 functions
#undef EXPLICIT_ASPACES

#define GENIA(A,T) \
ATTR void \
atomic_init(volatile A atomic_##T *p, T v) \
{ \
    __opencl_atomic_init((VOLATILE A atomic_##T *)p, v); \
}

#define GENSA(A,T) \
ATTR void \
atomic_store(volatile A atomic_##T *p, T v) \
{ \
    __opencl_atomic_store((VOLATILE A atomic_##T *)p, v, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR void \
atomic_store_explicit(volatile A atomic_##T *p, T v, memory_order o) \
{ \
    __opencl_atomic_store((VOLATILE A atomic_##T *)p, v, o, memory_scope_device); \
} \
 \
ATTR void \
atomic_store_explicit(volatile A atomic_##T *p, T v, memory_order o, memory_scope s) \
{ \
    __opencl_atomic_store((VOLATILE A atomic_##T *)p, v, o, s); \
}

#define GENLA(A,T) \
ATTR T \
atomic_load(volatile A atomic_##T *p) \
{ \
    return __opencl_atomic_load((VOLATILE A atomic_##T *)p, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR T \
atomic_load_explicit(volatile A atomic_##T *p, memory_order o) \
{ \
    return __opencl_atomic_load((VOLATILE A atomic_##T *)p, o, memory_scope_device); \
} \
 \
ATTR T \
atomic_load_explicit(volatile A atomic_##T *p, memory_order o, memory_scope s) \
{ \
    return __opencl_atomic_load((VOLATILE A atomic_##T *)p, o, s); \
}

#define GENXA(A,T) \
ATTR T \
atomic_exchange(volatile A atomic_##T *p, T v) \
{ \
    return RC_##T(__opencl_atomic_exchange((VOLATILE A AT_##T *)p, AC_##T(v), memory_order_seq_cst, memory_scope_device)); \
} \
 \
ATTR T \
atomic_exchange_explicit(volatile A atomic_##T *p, T v, memory_order o) \
{ \
    return RC_##T(__opencl_atomic_exchange((VOLATILE A AT_##T *)p, AC_##T(v), o, memory_scope_device)); \
} \
 \
ATTR T \
atomic_exchange_explicit(volatile A atomic_##T *p, T v, memory_order o, memory_scope s) \
{ \
    return RC_##T(__opencl_atomic_exchange((VOLATILE A AT_##T *)p, AC_##T(v), o, s)); \
}

#define GENCXAA(AP,AE,T,K) \
ATTR bool \
atomic_compare_exchange_##K(volatile AP atomic_##T *p, AE T *e, T d) \
{ \
    return __opencl_atomic_compare_exchange_##K((VOLATILE AP AT_##T *) p, (AE ET_##T *) e, AC_##T(d), memory_order_seq_cst, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR bool \
atomic_compare_exchange_##K##_explicit(volatile AP atomic_##T *p, AE T *e, T d, memory_order os, memory_order of) \
{ \
    return __opencl_atomic_compare_exchange_##K((VOLATILE AP AT_##T *)p, (AE ET_##T *)e, AC_##T(d), os, of, memory_scope_device); \
} \
 \
ATTR bool \
atomic_compare_exchange_##K##_explicit(volatile AP atomic_##T *p, AE T *e, T d, memory_order os, memory_order of, memory_scope s) \
{ \
    return __opencl_atomic_compare_exchange_##K((VOLATILE AP AT_##T *) p, (AE ET_##T *)e, AC_##T(d), os, of, s); \
}

#if defined EXPLICIT_ASPACES
#define GENCXA(A,T,K) \
    GENCXAA(A,__global,T,K) \
    GENCXAA(A,__local,T,K) \
    GENCXAA(A,__private,T,K) \
    GENCXAA(A,,T,K)
#else
#define GENCXA(A,T,K) GENCXAA(A,,T,K)
#endif

#define GENFOA(A,T,O) \
ATTR T \
atomic_fetch_##O(volatile A atomic_##T *p, T v) \
{ \
    return RC_##T(__opencl_atomic_fetch_##O((VOLATILE A AT_##T *)p, AC_##T(v), memory_order_seq_cst, memory_scope_device)); \
} \
 \
ATTR T \
atomic_fetch_##O##_explicit(volatile A atomic_##T *p, T v, memory_order o) \
{ \
    return RC_##T(__opencl_atomic_fetch_##O((VOLATILE A AT_##T *)p, AC_##T(v), o, memory_scope_device)); \
} \
 \
ATTR T \
atomic_fetch_##O##_explicit(volatile A atomic_##T *p, T v, memory_order o, memory_scope s) \
{ \
    return RC_##T(__opencl_atomic_fetch_##O((VOLATILE A AT_##T *) p, AC_##T(v), o, s)); \
}

#define CXA(A,T) \
    GENCXA(A,T,strong) \
    GENCXA(A,T,weak)

#define FOA(A,T) \
    GENFOA(A,T,add) \
    GENFOA(A,T,sub) \
    GENFOA(A,T,or) \
    GENFOA(A,T,xor) \
    GENFOA(A,T,and) \
    GENFOA(A,T,min) \
    GENFOA(A,T,max) \

#define ALLIA(A,F) \
    F(A,int) \
    F(A,uint) \
    F(A,long) \
    F(A,ulong)

#define ALLA(A,F) \
    ALLIA(A,F) \
    F(A,float) \
    F(A,double)

#if defined EXPLICIT_ASPACES
#define ALLI(F) \
    ALLIA(__global, F) \
    ALLIA(__local, F) \
    ALLIA(, F)
#else
#define ALLI(F) ALLIA(, F)
#endif

#if defined EXPLICIT_ASPACES
#define ALL(F) \
    ALLA(__global,F) \
    ALLA(__local, F) \
    ALLA(, F)
#else
#define ALL(F) ALLA(, F)
#endif

ALL(GENIA)
ALL(GENLA)
ALL(GENSA)
ALL(GENXA)
ALL(CXA)
ALLI(FOA)

// These are needed for uintptr_t
#define UIP(A) \
ATTR ulong \
atomic_fetch_add(volatile A atomic_ulong *p, long v) \
{ \
    return __opencl_atomic_fetch_add((VOLATILE A atomic_ulong *)p, (ulong)v, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR ulong \
atomic_fetch_add_explicit(volatile A atomic_ulong *p, long v, memory_order o) \
{ \
    return __opencl_atomic_fetch_add((VOLATILE A atomic_ulong *)p, (ulong)v, o, memory_scope_device); \
} \
 \
ATTR ulong \
atomic_fetch_add_explicit(volatile A atomic_ulong *p, long v, memory_order o, memory_scope s) \
{ \
    return __opencl_atomic_fetch_add((VOLATILE A atomic_ulong *)p, (ulong)v, o, s); \
} \
 \
ATTR ulong \
atomic_fetch_sub(volatile A atomic_ulong *p, long v) \
{ \
    return __opencl_atomic_fetch_sub((VOLATILE A atomic_ulong *)p, (ulong)v, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR ulong \
atomic_fetch_sub_explicit(volatile A atomic_ulong *p, long v, memory_order o) \
{ \
    return __opencl_atomic_fetch_sub((VOLATILE A atomic_ulong *)p, (ulong)v, o, memory_scope_device); \
} \
 \
ATTR ulong \
atomic_fetch_sub_explicit(volatile A atomic_ulong *p, long v, memory_order o, memory_scope s) \
{ \
    return __opencl_atomic_fetch_sub((VOLATILE A atomic_ulong *)p, (ulong)v, o, s); \
}

#if defined EXPLICIT_ASPACES
UIP(__global)
UIP(__local)
#endif
UIP()

// flag functions
#define FLG(A) \
ATTR bool \
atomic_flag_test_and_set(volatile A atomic_flag *p) \
{ \
    return __opencl_atomic_exchange((VOLATILE A atomic_int *)p, 1, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR bool \
atomic_flag_test_and_set_explicit(volatile A atomic_flag *p, memory_order o) \
{ \
    return __opencl_atomic_exchange((VOLATILE A atomic_int *)p, 1, o, memory_scope_device); \
} \
 \
ATTR bool \
atomic_flag_test_and_set_explicit(volatile A atomic_flag *p, memory_order o, memory_scope s) \
{ \
    return __opencl_atomic_exchange((VOLATILE A atomic_int *)p, 1, o, s); \
} \
 \
ATTR void \
atomic_flag_clear(volatile A atomic_flag *p) \
{ \
    __opencl_atomic_store((VOLATILE A atomic_int *)p, 0, memory_order_seq_cst, memory_scope_device); \
} \
 \
ATTR void \
atomic_flag_clear_explicit(volatile A atomic_flag *p, memory_order o) \
{ \
    __opencl_atomic_store((VOLATILE A atomic_int *)p, 0, o, memory_scope_device); \
} \
 \
ATTR void \
atomic_flag_clear_explicit(volatile A atomic_flag *p, memory_order o, memory_scope s) \
{ \
    __opencl_atomic_store((VOLATILE A atomic_int *)p, 0, o, s); \
} \

#if defined EXPLICIT_ASPACES
FLG(__global)
FLG(__local)
#endif
FLG()

