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

#if defined(_POSIX_THREADS) && _POSIX_THREADS > 0
#  define LIBCXXABI_SINGLE_THREADED 0
#else
#  define LIBCXXABI_SINGLE_THREADED 1
#endif

#endif // LIBCXXABI_CONFIG_H
