//===-- include/flang/Runtime/freestanding-tools.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_
#define FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/c-or-cpp.h"
#include <algorithm>
#include <cctype>
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

#if !defined(STD_REALLOC_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_REALLOC_UNSUPPORTED 1
#endif

#if !defined(STD_MEMCHR_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_MEMCHR_UNSUPPORTED 1
#endif

#if !defined(STD_STRCPY_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_STRCPY_UNSUPPORTED 1
#endif

#if !defined(STD_STRCMP_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_STRCMP_UNSUPPORTED 1
#endif

#if !defined(STD_TOUPPER_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_TOUPPER_UNSUPPORTED 1
#endif

#if defined(OMP_OFFLOAD_BUILD) && defined(OMP_NOHOST_BUILD) && \
    defined(__clang__)
#define STD_FILL_N_UNSUPPORTED 1
#define STD_MEMSET_USE_BUILTIN 1
#define STD_MEMSET_UNSUPPORTED 1
#define STD_MEMCPY_USE_BUILTIN 1
#define STD_MEMCPY_UNSUPPORTED 1
#define STD_MEMMOVE_UNSUPPORTED 1
#define STD_STRLEN_UNSUPPORTED 1
#define STD_MEMCMP_UNSUPPORTED 1
#define STD_REALLOC_UNSUPPORTED 1
#define STD_MEMCHR_UNSUPPORTED 1
#define STD_STRCPY_UNSUPPORTED 1
#define STD_STRCMP_UNSUPPORTED 1
#define STD_TOUPPER_UNSUPPORTED 1
#define STD_ABORT_USE_BUILTIN 1
#define STD_ABORT_UNSUPPORTED 1
#endif

namespace Fortran::runtime {

#if STD_FILL_N_UNSUPPORTED
// Provides alternative implementation for std::fill_n(), if
// it is not supported.
template <typename A, typename B>
static inline RT_API_ATTRS std::enable_if_t<std::is_convertible_v<B, A>, void>
fill_n(A *start, std::size_t count, const B &value) {
  for (std::size_t j{0}; j < count; ++j) {
    start[j] = value;
  }
}
#else // !STD_FILL_N_UNSUPPORTED
using std::fill_n;
#endif // !STD_FILL_N_UNSUPPORTED

#if STD_MEMSET_USE_BUILTIN
static inline RT_API_ATTRS void memset(
    void *dest, unsigned char value, std::size_t count) {
  __builtin_memset(dest, value, count);
}
#elif STD_MEMSET_UNSUPPORTED
static inline RT_API_ATTRS void memset(
    void *dest, unsigned char value, std::size_t count) {
  char *to{reinterpret_cast<char *>(dest)};
  while (count--) {
    *to++ = value;
  }
  return;
}
#else
using std::memset;
#endif

#if STD_MEMCPY_USE_BUILTIN
static inline RT_API_ATTRS void memcpy(
    void *dest, const void *src, std::size_t count) {
  __builtin_memcpy(dest, src, count);
}
#elif STD_MEMCPY_UNSUPPORTED
static inline RT_API_ATTRS void memcpy(
    void *dest, const void *src, std::size_t count) {
  char *to{reinterpret_cast<char *>(dest)};
  const char *from{reinterpret_cast<const char *>(src)};
  if (to == from) {
    return;
  }
  while (count--) {
    *to++ = *from++;
  }
}
#else
using std::memcpy;
#endif

#if STD_MEMMOVE_USE_BUILTIN
static inline RT_API_ATTRS void memmove(
    void *dest, const void *src, std::size_t count) {
  __builtin_memmove(dest, src, count);
}
#elif STD_MEMMOVE_UNSUPPORTED
// Provides alternative implementation for std::memmove(), if
// it is not supported.
static inline RT_API_ATTRS void *memmove(
    void *dest, const void *src, std::size_t count) {
  char *to{reinterpret_cast<char *>(dest)};
  const char *from{reinterpret_cast<const char *>(src)};

  if (to == from) {
    return dest;
  }
  if (to + count <= from || from + count <= to) {
    memcpy(dest, src, count);
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
  return dest;
}
#else // !STD_MEMMOVE_UNSUPPORTED
using std::memmove;
#endif // !STD_MEMMOVE_UNSUPPORTED

using MemmoveFct = void *(*)(void *, const void *, std::size_t);

#ifdef RT_DEVICE_COMPILATION
[[maybe_unused]] static RT_API_ATTRS void *MemmoveWrapper(
    void *dest, const void *src, std::size_t count) {
  return Fortran::runtime::memmove(dest, src, count);
}
#endif

#if STD_STRLEN_USE_BUILTIN
static inline RT_API_ATTRS std::size_t strlen(const char *str) {
  return __builtin_strlen(str);
}
#elif STD_STRLEN_UNSUPPORTED
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

#if STD_REALLOC_UNSUPPORTED
static inline RT_API_ATTRS void *realloc(void *ptr, std::size_t newByteSize) {
  // Return nullptr and let the callers assert that.
  // TODO: we can provide a straightforward implementation
  // via malloc/memcpy/free.
  return nullptr;
}
#else // !STD_REALLOC_UNSUPPORTED
using std::realloc;
#endif // !STD_REALLOC_UNSUPPORTED

#if STD_MEMCHR_UNSUPPORTED
// Provides alternative implementation for std::memchr(), if
// it is not supported.
static inline RT_API_ATTRS const void *memchr(
    const void *ptr, int ch, std::size_t count) {
  auto buf{reinterpret_cast<const unsigned char *>(ptr)};
  auto c{static_cast<unsigned char>(ch)};
  for (; count--; ++buf) {
    if (*buf == c) {
      return buf;
    }
  }
  return nullptr;
}
#else // !STD_MEMCMP_UNSUPPORTED
using std::memchr;
#endif // !STD_MEMCMP_UNSUPPORTED

#if STD_STRCPY_UNSUPPORTED
// Provides alternative implementation for std::strcpy(), if
// it is not supported.
static inline RT_API_ATTRS char *strcpy(char *dest, const char *src) {
  char *result{dest};
  do {
    *dest++ = *src;
  } while (*src++ != '\0');
  return result;
}
#else // !STD_STRCPY_UNSUPPORTED
using std::strcpy;
#endif // !STD_STRCPY_UNSUPPORTED

#if STD_STRCMP_UNSUPPORTED
// Provides alternative implementation for std::strcmp(), if
// it is not supported.
static inline RT_API_ATTRS int strcmp(const char *lhs, const char *rhs) {
  while (*lhs != '\0' && *lhs == *rhs) {
    ++lhs;
    ++rhs;
  }
  return static_cast<unsigned char>(*lhs) - static_cast<unsigned char>(*rhs);
}
#else // !STD_STRCMP_UNSUPPORTED
using std::strcmp;
#endif // !STD_STRCMP_UNSUPPORTED

#if STD_TOUPPER_UNSUPPORTED
// Provides alternative implementation for std::toupper(), if
// it is not supported.
static inline RT_API_ATTRS int toupper(int ch) {
  if (ch >= 'a' && ch <= 'z') {
    return ch - 'a' + 'A';
  }
  return ch;
}
#else // !STD_TOUPPER_UNSUPPORTED
using std::toupper;
#endif // !STD_TOUPPER_UNSUPPORTED

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_FREESTANDING_TOOLS_H_
