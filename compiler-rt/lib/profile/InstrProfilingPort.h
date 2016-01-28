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
#elif __GNUC__
#define COMPILER_RT_ALIGNAS(x) __attribute__((aligned(x)))
#define COMPILER_RT_VISIBILITY __attribute__((visibility("hidden")))
#define COMPILER_RT_WEAK __attribute__((weak))
#endif

#define COMPILER_RT_SECTION(Sect) __attribute__((section(Sect)))

#if COMPILER_RT_HAS_ATOMICS == 1
#ifdef _MSC_VER
#include <windows.h>
#define snprintf _snprintf
#if defined(_WIN64)
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  (InterlockedCompareExchange64((LONGLONG volatile *)Ptr, (LONGLONG)NewV,      \
                                (LONGLONG)OldV) == (LONGLONG)OldV)
#else /* !defined(_WIN64) */
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  (InterlockedCompareExchange((LONG volatile *)Ptr, (LONG)NewV, (LONG)OldV) == \
   (LONG)OldV)
#endif
#else /* !defined(_MSC_VER) */
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  __sync_bool_compare_and_swap(Ptr, OldV, NewV)
#endif
#else /* COMPILER_RT_HAS_ATOMICS != 1 */
#define COMPILER_RT_BOOL_CMPXCHG(Ptr, OldV, NewV)                              \
  BoolCmpXchg((void **)Ptr, OldV, NewV)
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
