//===-- runtime/freestanding-tools.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_
#define FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_

#include "flang/Runtime/api-attrs.h"
#include "flang/Runtime/c-or-cpp.h"
#include <algorithm>
#include <cstring>

// The file defines a set of utilities/classes that might be
// used to get reduce the dependency on external libraries (e.g. libstdc++).

#if !defined(STD_FILL_N_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_FILL_N_UNSUPPORTED 1
#endif

#if !defined(STD_MEMMOVE_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_MEMMOVE_UNSUPPORTED 1
#endif

#if !defined(STD_STRLEN_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_STRLEN_UNSUPPORTED 1
#endif

#if !defined(STD_MEMCMP_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_MEMCMP_UNSUPPORTED 1
#endif

namespace Fortran::runtime {

#if STD_FILL_N_UNSUPPORTED
// Provides alternative implementation for std::fill_n(), if
// it is not supported.
template <typename A>
static inline RT_API_ATTRS void fill_n(
    A *start, std::size_t count, const A &value) {
  for (std::size_t j{0}; j < count; ++j) {
    start[j] = value;
  }
}
#else // !STD_FILL_N_UNSUPPORTED
using std::fill_n;
#endif // !STD_FILL_N_UNSUPPORTED

#if STD_MEMMOVE_UNSUPPORTED
// Provides alternative implementation for std::memmove(), if
// it is not supported.
static inline RT_API_ATTRS void memmove(
    void *dest, const void *src, std::size_t count) {
  char *to{reinterpret_cast<char *>(dest)};
  const char *from{reinterpret_cast<const char *>(src)};

  if (to == from) {
    return;
  }
  if (to + count <= from || from + count <= to) {
    std::memcpy(dest, src, count);
  } else if (to < from) {
    while (count--) {
      *to++ = *from++;
    }
  } else {
    to += count;
    from += count;
    while (count--) {
      *--to = *--from;
    }
  }
}
#else // !STD_MEMMOVE_UNSUPPORTED
using std::memmove;
#endif // !STD_MEMMOVE_UNSUPPORTED

#if STD_STRLEN_UNSUPPORTED
// Provides alternative implementation for std::strlen(), if
// it is not supported.
static inline RT_API_ATTRS std::size_t strlen(const char *str) {
  if (!str) {
    // Return 0 for nullptr.
    return 0;
  }
  const char *end = str;
  for (; *end != '\0'; ++end)
    ;
  return end - str;
}
#else // !STD_STRLEN_UNSUPPORTED
using std::strlen;
#endif // !STD_STRLEN_UNSUPPORTED

#if STD_MEMCMP_UNSUPPORTED
// Provides alternative implementation for std::memcmp(), if
// it is not supported.
static inline RT_API_ATTRS int memcmp(
    const void *RESTRICT lhs, const void *RESTRICT rhs, std::size_t count) {
  auto m1{reinterpret_cast<const unsigned char *>(lhs)};
  auto m2{reinterpret_cast<const unsigned char *>(rhs)};
  for (; count--; ++m1, ++m2) {
    int diff = *m1 - *m2;
    if (diff != 0) {
      return diff;
    }
  }
  return 0;
}
#else // !STD_MEMCMP_UNSUPPORTED
using std::memcmp;
#endif // !STD_MEMCMP_UNSUPPORTED

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_
