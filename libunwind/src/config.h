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
  #include <Availability.h>
  #ifdef __cplusplus
    extern "C" {
  #endif
    void __assert_rtn(const char *, const char *, int, const char *)
                                                      __attribute__((noreturn));
  #ifdef __cplusplus
    }
  #endif

  #define _LIBUNWIND_BUILD_ZERO_COST_APIS (defined(__i386__) || \
                                           defined(__x86_64__) || \
                                           defined(__arm64__))
  #define _LIBUNWIND_BUILD_SJLJ_APIS      defined(__arm__)
  #define _LIBUNWIND_SUPPORT_FRAME_APIS   (defined(__i386__) || \
                                           defined(__x86_64__))
  #define _LIBUNWIND_EXPORT               __attribute__((visibility("default")))
  #define _LIBUNWIND_HIDDEN               __attribute__((visibility("hidden")))
  #define _LIBUNWIND_LOG(msg, ...) fprintf(stderr, "libuwind: " msg, __VA_ARGS__)
  #define _LIBUNWIND_ABORT(msg) __assert_rtn(__func__, __FILE__, __LINE__, msg)

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
  #include <stdlib.h>

  static inline void assert_rtn(const char* func, const char* file, int line, const char* msg)  __attribute__ ((noreturn));
  static inline void assert_rtn(const char* func, const char* file, int line, const char* msg) {
    fprintf(stderr, "libunwind: %s %s:%d - %s\n",  func, file, line, msg);
    assert(false);
    abort();
  }

  #define _LIBUNWIND_BUILD_ZERO_COST_APIS (defined(__i386__) || \
                                           defined(__x86_64__) || \
                                           defined(__arm__))
  #define _LIBUNWIND_BUILD_SJLJ_APIS      0
  #define _LIBUNWIND_SUPPORT_FRAME_APIS   (defined(__i386__) || \
                                           defined(__x86_64__))
  #define _LIBUNWIND_EXPORT               __attribute__((visibility("default")))
  #define _LIBUNWIND_HIDDEN               __attribute__((visibility("hidden")))
  #define _LIBUNWIND_LOG(msg, ...) fprintf(stderr, "libuwind: " msg, __VA_ARGS__)
  #define _LIBUNWIND_ABORT(msg) assert_rtn(__func__, __FILE__, __LINE__, msg)

  #define _LIBUNWIND_SUPPORT_COMPACT_UNWIND 0
  #define _LIBUNWIND_SUPPORT_DWARF_UNWIND !defined(__arm__) || \
                                          defined(__ARM_DWARF_EH__)
  #define _LIBUNWIND_SUPPORT_DWARF_INDEX _LIBUNWIND_SUPPORT_DWARF_UNWIND
#endif


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


#endif // LIBUNWIND_CONFIG_H
