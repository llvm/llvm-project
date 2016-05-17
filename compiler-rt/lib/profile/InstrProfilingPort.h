/*===- InstrProfilingPort.h- Support library for PGO instrumentation ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILING_PORT_H_
#define PROFILE_INSTRPROFILING_PORT_H_

#ifdef _MSC_VER
#define COMPILER_RT_ALIGNAS(x) __declspec(align(x))
#define COMPILER_RT_VISIBILITY
#define COMPILER_RT_WEAK __declspec(selectany)
#define COMPILER_RT_ALLOCA _alloca
#elif __GNUC__
#define COMPILER_RT_ALIGNAS(x) __attribute__((aligned(x)))
#define COMPILER_RT_VISIBILITY __attribute__((visibility("hidden")))
#define COMPILER_RT_WEAK __attribute__((weak))
#define COMPILER_RT_ALLOCA __builtin_alloca
#endif

#define COMPILER_RT_SECTION(Sect) __attribute__((section(Sect)))

#define COMPILER_RT_MAX_HOSTLEN 128
#ifdef _MSC_VER
#define COMPILER_RT_GETHOSTNAME(Name, Len) gethostname(Name, Len)
#elif defined(__ORBIS__)
#define COMPILER_RT_GETHOSTNAME(Name, Len) (-1)
#else
#define COMPILER_RT_GETHOSTNAME(Name, Len) lprofGetHostName(Name, Len)
#define COMPILER_RT_HAS_UNAME 1
#endif

#if COMPILER_RT_HAS_ATOMICS == 1
#ifdef _MSC_VER
#include <windows.h>
#if _MSC_VER < 1900
#define snprintf _snprintf
#endif
#if defined(_WIN64)
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  (InterlockedCompareExchange64((LONGLONG volatile *)Ptr, (LONGLONG)NewV,      \
                                (LONGLONG)OldV) == (LONGLONG)OldV)
#define COMPILER_RT_PTR_FETCH_ADD(DomType, PtrVar, PtrIncr)                    \
  (DomType *)InterlockedExchangeAdd64((LONGLONG volatile *)&PtrVar,            \
                                      (LONGLONG)sizeof(DomType) * PtrIncr)
#else /* !defined(_WIN64) */
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  (InterlockedCompareExchange((LONG volatile *)Ptr, (LONG)NewV, (LONG)OldV) == \
   (LONG)OldV)
#define COMPILER_RT_PTR_FETCH_ADD(DomType, PtrVar, PtrIncr)                    \
  (DomType *)InterlockedExchangeAdd((LONG volatile *)&PtrVar,                  \
                                    (LONG)sizeof(DomType) * PtrIncr)
#endif
#else /* !defined(_MSC_VER) */
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  __sync_bool_compare_and_swap(Ptr, OldV, NewV)
#define COMPILER_RT_PTR_FETCH_ADD(DomType, PtrVar, PtrIncr)                    \
  (DomType *)__sync_fetch_and_add((long *)&PtrVar, sizeof(DomType) * PtrIncr)
#endif
#else /* COMPILER_RT_HAS_ATOMICS != 1 */
#include "InstrProfilingUtil.h"
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  lprofBoolCmpXchg((void **)Ptr, OldV, NewV)
#define COMPILER_RT_PTR_FETCH_ADD(DomType, PtrVar, PtrIncr)                    \
  (DomType *)lprofPtrFetchAdd((void **)&PtrVar, sizeof(DomType) * PtrIncr)
#endif

#define PROF_ERR(Format, ...)                                                  \
  if (GetEnvHook && GetEnvHook("LLVM_PROFILE_VERBOSE_ERRORS"))                 \
    fprintf(stderr, Format, __VA_ARGS__);

#if defined(__FreeBSD__)

#include <inttypes.h>
#include <sys/types.h>

#else /* defined(__FreeBSD__) */

#include <inttypes.h>
#include <stdint.h>

#endif /* defined(__FreeBSD__) && defined(__i386__) */

#endif /* PROFILE_INSTRPROFILING_PORT_H_ */
