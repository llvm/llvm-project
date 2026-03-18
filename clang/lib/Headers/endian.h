//===-- Definition of macros from endian.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_ENDIAN_H
#define __CLANG_ENDIAN_H

// If the system has an endian.h, let's use that instead.
#if __STDC_HOSTED__ && __has_include_next(<endian.h>)
#include_next <endian.h>
#else

#include <stdint.h>

// Implementation taken from llvm libc endian-macros.h.
#ifdef __cplusplus
#define __CLANG_ENDIAN_CAST(cast, type, value) (cast<type>(value))
#else
#define __CLANG_ENDIAN_CAST(cast, type, value) ((type)(value))
#endif

#define LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
#define BIG_ENDIAN __ORDER_BIG_ENDIAN__
#define BYTE_ORDER __BYTE_ORDER__

#if BYTE_ORDER == LITTLE_ENDIAN

#define htobe16(x)                                                             \
  __builtin_bswap16(__CLANG_ENDIAN_CAST(static_cast, uint16_t, x))
#define htobe32(x)                                                             \
  __builtin_bswap32(__CLANG_ENDIAN_CAST(static_cast, uint32_t, x))
#define htobe64(x)                                                             \
  __builtin_bswap64(__CLANG_ENDIAN_CAST(static_cast, uint64_t, x))
#define htole16(x) __CLANG_ENDIAN_CAST(static_cast, uint16_t, x)
#define htole32(x) __CLANG_ENDIAN_CAST(static_cast, uint32_t, x)
#define htole64(x) __CLANG_ENDIAN_CAST(static_cast, uint64_t, x)
#define be16toh(x)                                                             \
  __builtin_bswap16(__CLANG_ENDIAN_CAST(static_cast, uint16_t, x))
#define be32toh(x)                                                             \
  __builtin_bswap32(__CLANG_ENDIAN_CAST(static_cast, uint32_t, x))
#define be64toh(x)                                                             \
  __builtin_bswap64(__CLANG_ENDIAN_CAST(static_cast, uint64_t, x))
#define le16toh(x) __CLANG_ENDIAN_CAST(static_cast, uint16_t, x)
#define le32toh(x) __CLANG_ENDIAN_CAST(static_cast, uint32_t, x)
#define le64toh(x) __CLANG_ENDIAN_CAST(static_cast, uint64_t, x)

#else

#define htobe16(x) __CLANG_ENDIAN_CAST(static_cast, uint16_t, x)
#define htobe32(x) __CLANG_ENDIAN_CAST(static_cast, uint32_t, x)
#define htobe64(x) __CLANG_ENDIAN_CAST(static_cast, uint64_t, x)
#define htole16(x)                                                             \
  __builtin_bswap16(__CLANG_ENDIAN_CAST(static_cast, uint16_t, x))
#define htole32(x)                                                             \
  __builtin_bswap32(__CLANG_ENDIAN_CAST(static_cast, uint32_t, x))
#define htole64(x)                                                             \
  __builtin_bswap64(__CLANG_ENDIAN_CAST(static_cast, uint64_t, x))
#define be16toh(x) __CLANG_ENDIAN_CAST(static_cast, uint16_t, x)
#define be32toh(x) __CLANG_ENDIAN_CAST(static_cast, uint32_t, x)
#define be64toh(x) __CLANG_ENDIAN_CAST(static_cast, uint64_t, x)
#define le16toh(x)                                                             \
  __builtin_bswap16(__CLANG_ENDIAN_CAST(static_cast, uint16_t, x))
#define le32toh(x)                                                             \
  __builtin_bswap32(__CLANG_ENDIAN_CAST(static_cast, uint32_t, x))
#define le64toh(x)                                                             \
  __builtin_bswap64(__CLANG_ENDIAN_CAST(static_cast, uint64_t, x))

#endif
#endif // __has_include_next
#endif // __CLANG_ENDIAN_H
