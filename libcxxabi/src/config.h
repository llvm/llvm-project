//===----------------------------- config.h -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
//  Defines macros used within the libc++abi project.
//
//===----------------------------------------------------------------------===//


#ifndef LIBCXXABI_CONFIG_H
#define LIBCXXABI_CONFIG_H

#include <unistd.h>

#ifndef __has_attribute
  #define __has_attribute(x) 0
#endif

// Configure inline visibility attributes
#if defined(_WIN32)
 #if defined(_MSC_VER) && !defined(__clang__)
  // Using Microsoft Visual C++ compiler
  #define _LIBCXXABI_INLINE_VISIBILITY __forceinline
 #else
  #if __has_attribute(__internal_linkage__)
   #define _LIBCXXABI_INLINE_VISIBILITY __attribute__ ((__internal_linkage__, __always_inline__))
  #else
   #define _LIBCXXABI_INLINE_VISIBILITY __attribute__ ((__always_inline__))
  #endif
 #endif
#else
 #if __has_attribute(__internal_linkage__)
  #define _LIBCXXABI_INLINE_VISIBILITY __attribute__ ((__internal_linkage__, __always_inline__))
 #else
  #define _LIBCXXABI_INLINE_VISIBILITY __attribute__ ((__visibility__("hidden"), __always_inline__))
 #endif
#endif

// Try and deduce a threading api if one has not been explicitly set.
#if !defined(_LIBCXXABI_HAS_NO_THREADS) && \
    !defined(_LIBCXXABI_USE_THREAD_API_PTHREAD)
  #if defined(_POSIX_THREADS) && _POSIX_THREADS >= 0
    #define _LIBCXXABI_USE_THREAD_API_PTHREAD
  #else
    #error "No thread API"
  #endif
#endif

// Set this in the CXXFLAGS when you need it, because otherwise we'd have to
// #if !defined(__linux__) && !defined(__APPLE__) && ...
// and so-on for *every* platform.
#ifndef LIBCXXABI_BAREMETAL
#  define LIBCXXABI_BAREMETAL 0
#endif

#endif // LIBCXXABI_CONFIG_H
