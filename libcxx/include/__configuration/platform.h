// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_PLATFORM_H
#define _LIBCPP___CONFIGURATION_PLATFORM_H

#include <__config_site>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#if defined(__ELF__)
#  define _LIBCPP_OBJECT_FORMAT_ELF 1
#elif defined(__MACH__)
#  define _LIBCPP_OBJECT_FORMAT_MACHO 1
#elif defined(_WIN32)
#  define _LIBCPP_OBJECT_FORMAT_COFF 1
#elif defined(__wasm__)
#  define _LIBCPP_OBJECT_FORMAT_WASM 1
#elif defined(_AIX)
#  define _LIBCPP_OBJECT_FORMAT_XCOFF 1
#else
// ... add new file formats here ...
#endif

#if defined(__MVS__)
#  include <features.h> // for __NATIVE_ASCII_F
#endif

// Need to detect which libc we're using if we're on Linux.
#if (defined(__linux__) || defined(__AMDGPU__) || defined(__NVPTX__)) && __has_include(<features.h>)
#  include <features.h>
#  if defined(__GLIBC_PREREQ)
#    define _LIBCPP_GLIBC_PREREQ(a, b) __GLIBC_PREREQ(a, b)
#  else
#    define _LIBCPP_GLIBC_PREREQ(a, b) 0
#  endif // defined(__GLIBC_PREREQ)
#else
#  define _LIBCPP_GLIBC_PREREQ(a, b) 0
#endif

#ifndef __BYTE_ORDER__
#  error                                                                                                               \
      "Your compiler doesn't seem to define __BYTE_ORDER__, which is required by libc++ to know the endianness of your target platform"
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define _LIBCPP_LITTLE_ENDIAN
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define _LIBCPP_BIG_ENDIAN
#endif // __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__

// Libc++ supports various implementations of std::random_device.
//
// _LIBCPP_USING_DEV_RANDOM
//      Read entropy from the given file, by default `/dev/urandom`.
//      If a token is provided, it is assumed to be the path to a file
//      to read entropy from. This is the default behavior if nothing
//      else is specified. This implementation requires storing state
//      inside `std::random_device`.
//
// _LIBCPP_USING_ARC4_RANDOM
//      Use arc4random(). This allows obtaining random data even when
//      using sandboxing mechanisms. On some platforms like Apple, this
//      is the recommended source of entropy for user-space programs.
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
//
// _LIBCPP_USING_GETENTROPY
//      Use getentropy().
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
//
// _LIBCPP_USING_FUCHSIA_CPRNG
//      Use Fuchsia's zx_cprng_draw() system call, which is specified to
//      deliver high-quality entropy and cannot fail.
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
//
// _LIBCPP_USING_WIN32_RANDOM
//      Use rand_s(), for use on Windows.
//      When this option is used, the token passed to `std::random_device`'s
//      constructor *must* be "/dev/urandom" -- anything else is an error.
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#  define _LIBCPP_USING_ARC4_RANDOM
#elif defined(__wasi__) || defined(__EMSCRIPTEN__)
#  define _LIBCPP_USING_GETENTROPY
#elif defined(__Fuchsia__)
#  define _LIBCPP_USING_FUCHSIA_CPRNG
#elif defined(_LIBCPP_WIN32API)
#  define _LIBCPP_USING_WIN32_RANDOM
#else
#  define _LIBCPP_USING_DEV_RANDOM
#endif

// Thread API
// clang-format off
#if _LIBCPP_HAS_THREADS &&                                                                                             \
    !_LIBCPP_HAS_THREAD_API_PTHREAD &&                                                                                 \
    !_LIBCPP_HAS_THREAD_API_WIN32 &&                                                                                   \
    !_LIBCPP_HAS_THREAD_API_EXTERNAL &&                                                                                \
    !_LIBCPP_HAS_THREAD_API_C11

#  if defined(__FreeBSD__) ||                                                                                          \
      defined(__wasi__) ||                                                                                             \
      defined(__NetBSD__) ||                                                                                           \
      defined(__OpenBSD__) ||                                                                                          \
      defined(__NuttX__) ||                                                                                            \
      defined(__linux__) ||                                                                                            \
      defined(__GNU__) ||                                                                                              \
      defined(__APPLE__) ||                                                                                            \
      defined(__MVS__) ||                                                                                              \
      defined(_AIX) ||                                                                                                 \
      defined(__EMSCRIPTEN__)
// clang-format on
#    undef _LIBCPP_HAS_THREAD_API_PTHREAD
#    define _LIBCPP_HAS_THREAD_API_PTHREAD 1
#  elif defined(__Fuchsia__)
// TODO(44575): Switch to C11 thread API when possible.
#    undef _LIBCPP_HAS_THREAD_API_PTHREAD
#    define _LIBCPP_HAS_THREAD_API_PTHREAD 1
#  elif defined(_LIBCPP_WIN32API)
#    undef _LIBCPP_HAS_THREAD_API_WIN32
#    define _LIBCPP_HAS_THREAD_API_WIN32 1
#  else
#    error "No thread API"
#  endif // _LIBCPP_HAS_THREAD_API
#endif   // _LIBCPP_HAS_THREADS

#if !_LIBCPP_HAS_THREAD_API_PTHREAD
#  define _LIBCPP_HAS_COND_CLOCKWAIT 0
#elif (defined(__ANDROID__) && __ANDROID_API__ >= 30) || _LIBCPP_GLIBC_PREREQ(2, 30)
#  define _LIBCPP_HAS_COND_CLOCKWAIT 1
#else
#  define _LIBCPP_HAS_COND_CLOCKWAIT 0
#endif

#if !_LIBCPP_HAS_THREADS && _LIBCPP_HAS_THREAD_API_PTHREAD
#  error _LIBCPP_HAS_THREAD_API_PTHREAD may only be true when _LIBCPP_HAS_THREADS is true.
#endif

#if !_LIBCPP_HAS_THREADS && _LIBCPP_HAS_THREAD_API_EXTERNAL
#  error _LIBCPP_HAS_THREAD_API_EXTERNAL may only be true when _LIBCPP_HAS_THREADS is true.
#endif

#if !_LIBCPP_HAS_THREADS && _LIBCPP_HAS_THREAD_API_C11
#  error _LIBCPP_HAS_THREAD_API_C11 may only be true when _LIBCPP_HAS_THREADS is true.
#endif

#if !_LIBCPP_HAS_MONOTONIC_CLOCK && _LIBCPP_HAS_THREADS
#  error _LIBCPP_HAS_MONOTONIC_CLOCK may only be false when _LIBCPP_HAS_THREADS is false.
#endif

#if _LIBCPP_HAS_THREADS && !defined(__STDCPP_THREADS__)
#  define __STDCPP_THREADS__ 1
#endif

// The glibc and Bionic implementation of pthreads implements
// pthread_mutex_destroy as nop for regular mutexes. Additionally, Win32
// mutexes have no destroy mechanism.
//
// This optimization can't be performed on Apple platforms, where
// pthread_mutex_destroy can allow the kernel to release resources.
// See https://llvm.org/D64298 for details.
//
// TODO(EricWF): Enable this optimization on Bionic after speaking to their
//               respective stakeholders.
// clang-format off
#  if (_LIBCPP_HAS_THREAD_API_PTHREAD && defined(__GLIBC__)) ||                                                        \
      (_LIBCPP_HAS_THREAD_API_C11 && defined(__Fuchsia__)) ||                                                          \
       _LIBCPP_HAS_THREAD_API_WIN32
// clang-format on
#  define _LIBCPP_HAS_TRIVIAL_MUTEX_DESTRUCTION 1
#else
#  define _LIBCPP_HAS_TRIVIAL_MUTEX_DESTRUCTION 0
#endif

// Destroying a condvar is a nop on Windows.
//
// This optimization can't be performed on Apple platforms, where
// pthread_cond_destroy can allow the kernel to release resources.
// See https://llvm.org/D64298 for details.
//
// TODO(EricWF): This is potentially true for some pthread implementations
// as well.
#if (_LIBCPP_HAS_THREAD_API_C11 && defined(__Fuchsia__)) || _LIBCPP_HAS_THREAD_API_WIN32
#  define _LIBCPP_HAS_TRIVIAL_CONDVAR_DESTRUCTION 1
#else
#  define _LIBCPP_HAS_TRIVIAL_CONDVAR_DESTRUCTION 0
#endif

#if defined(__BIONIC__) || defined(__NuttX__) || defined(__Fuchsia__) || defined(__wasi__) || _LIBCPP_HAS_MUSL_LIBC || \
    defined(__OpenBSD__) || _LIBCPP_LIBC_LLVM_LIBC
#  define _LIBCPP_PROVIDES_DEFAULT_RUNE_TABLE
#endif

#endif // _LIBCPP___CONFIGURATION_PLATFORM_H
