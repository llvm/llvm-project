//===----------------------------- config.h -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
//  Defines macros used within libuwind project.
//
//===----------------------------------------------------------------------===//


#ifndef LIBUNWIND_CONFIG_H
#define LIBUNWIND_CONFIG_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Define static_assert() unless already defined by compiler.
#ifndef __has_feature
  #define __has_feature(__x) 0
#endif
#if !(__has_feature(cxx_static_assert)) && !defined(static_assert)
  #define static_assert(__b, __m) \
      extern int compile_time_assert_failed[ ( __b ) ? 1 : -1 ]  \
                                                  __attribute__( ( unused ) );
#endif

// Platform specific configuration defines.
#ifdef __APPLE__
  #if defined(FOR_DYLD)
    #define _LIBUNWIND_SUPPORT_COMPACT_UNWIND 1
    #define _LIBUNWIND_SUPPORT_DWARF_UNWIND   0
    #define _LIBUNWIND_SUPPORT_DWARF_INDEX    0
  #else
    #define _LIBUNWIND_SUPPORT_COMPACT_UNWIND 1
    #define _LIBUNWIND_SUPPORT_DWARF_UNWIND   1
    #define _LIBUNWIND_SUPPORT_DWARF_INDEX    0
  #endif
#else
  #if defined(__ARM_DWARF_EH__) || !defined(__arm__)
    #define _LIBUNWIND_SUPPORT_COMPACT_UNWIND 0
    #define _LIBUNWIND_SUPPORT_DWARF_UNWIND 1
    #define _LIBUNWIND_SUPPORT_DWARF_INDEX 1
  #else
    #define _LIBUNWIND_SUPPORT_COMPACT_UNWIND 0
    #define _LIBUNWIND_SUPPORT_DWARF_UNWIND 0
    #define _LIBUNWIND_SUPPORT_DWARF_INDEX 0
  #endif
#endif

// FIXME: these macros are not correct for COFF targets
#define _LIBUNWIND_EXPORT __attribute__((visibility("default")))
#define _LIBUNWIND_HIDDEN __attribute__((visibility("hidden")))

#if (defined(__APPLE__) && defined(__arm__)) || defined(__USING_SJLJ_EXCEPTIONS__)
#define _LIBUNWIND_BUILD_SJLJ_APIS 1
#else
#define _LIBUNWIND_BUILD_SJLJ_APIS 0
#endif

#if defined(__i386__) || defined(__x86_64__)
#define _LIBUNWIND_SUPPORT_FRAME_APIS 1
#else
#define _LIBUNWIND_SUPPORT_FRAME_APIS 0
#endif

#if defined(__i386__) || defined(__x86_64__) ||                                \
    (!defined(__APPLE__) && defined(__arm__)) ||                               \
    (defined(__arm64__) || defined(__aarch64__)) ||                            \
    (defined(__APPLE__) && defined(__mips__))
#define _LIBUNWIND_BUILD_ZERO_COST_APIS 1
#else
#define _LIBUNWIND_BUILD_ZERO_COST_APIS 0
#endif

#define _LIBUNWIND_ABORT(msg)                                                  \
  do {                                                                         \
    fprintf(stderr, "libunwind: %s %s:%d - %s\n", __func__, __FILE__,          \
            __LINE__, msg);                                                    \
    fflush(stderr);                                                            \
    abort();                                                                   \
  } while (0)
#define _LIBUNWIND_LOG(msg, ...) fprintf(stderr, "libuwind: " msg, __VA_ARGS__)

// Macros that define away in non-Debug builds
#ifdef NDEBUG
  #define _LIBUNWIND_DEBUG_LOG(msg, ...)
  #define _LIBUNWIND_TRACE_API(msg, ...)
  #define _LIBUNWIND_TRACING_UNWINDING 0
  #define _LIBUNWIND_TRACE_UNWINDING(msg, ...)
  #define _LIBUNWIND_LOG_NON_ZERO(x) x
#else
  #ifdef __cplusplus
    extern "C" {
  #endif
    extern  bool logAPIs();
    extern  bool logUnwinding();
  #ifdef __cplusplus
    }
  #endif
  #define _LIBUNWIND_DEBUG_LOG(msg, ...)  _LIBUNWIND_LOG(msg, __VA_ARGS__)
  #define _LIBUNWIND_LOG_NON_ZERO(x) \
            do { \
              int _err = x; \
              if ( _err != 0 ) \
                _LIBUNWIND_LOG("" #x "=%d in %s", _err, __FUNCTION__); \
             } while (0)
  #define _LIBUNWIND_TRACE_API(msg, ...) \
            do { \
              if ( logAPIs() ) _LIBUNWIND_LOG(msg, __VA_ARGS__); \
            } while(0)
  #define _LIBUNWIND_TRACE_UNWINDING(msg, ...) \
            do { \
              if ( logUnwinding() ) _LIBUNWIND_LOG(msg, __VA_ARGS__); \
            } while(0)
  #define _LIBUNWIND_TRACING_UNWINDING logUnwinding()
#endif

#ifdef __cplusplus
// Used to fit UnwindCursor and Registers_xxx types against unw_context_t /
// unw_cursor_t sized memory blocks.
#if defined(_LIBUNWIND_IS_NATIVE_ONLY)
# define COMP_OP ==
#else
# define COMP_OP <
#endif
template <typename _Type, typename _Mem>
struct check_fit {
  template <typename T>
  struct blk_count {
    static const size_t count =
      (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  };
  static const bool does_fit =
    (blk_count<_Type>::count COMP_OP blk_count<_Mem>::count);
};
#undef COMP_OP
#endif // __cplusplus

#endif // LIBUNWIND_CONFIG_H
