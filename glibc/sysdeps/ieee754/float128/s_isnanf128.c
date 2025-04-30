#include <shlib-compat.h>
#include <float128_private.h>
#if !IS_IN (libm)
#undef __isnanl
#define __isnanl __isnanf128_impl
#undef weak_alias
#define weak_alias(n,a)
#undef mathx_hidden_def
#define mathx_hidden_def(x)
#endif
#include "../ldbl-128/s_isnanl.c"
#if !IS_IN (libm)
#include <float128-abi.h>
hidden_ver (__isnanf128_impl, __isnanf128)
_weak_alias (__isnanf128_impl, isnanl)
versioned_symbol (libc, __isnanf128_impl, __isnanf128, GLIBC_2_34);
#if (SHLIB_COMPAT (libc, FLOAT128_VERSION_M, GLIBC_2_34))
strong_alias (__isnanf128_impl, __isnanf128_alias)
compat_symbol (libc, __isnanf128_alias, __isnanf128, FLOAT128_VERSION_M);
#endif
#endif
