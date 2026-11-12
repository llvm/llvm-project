//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains basic implementations of Scalable Matrix Extension (SME)
/// compatible memset and memchr functions to be used when their assembly-
/// optimized counterparts can't.
///
//===----------------------------------------------------------------------===//

#include <stddef.h>

extern void *__arm_sc_memset(void *dest, int c,
                             size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    destp[i] = c8;

  return dest;
}

extern const void *__arm_sc_memchr(const void *src, int c,
                                   size_t n) __arm_streaming_compatible {
  const unsigned char *srcp = (const unsigned char *)src;
  unsigned char c8 = (unsigned char)c;
  for (size_t i = 0; i < n; ++i)
    if (srcp[i] == c8)
      return &srcp[i];

  return NULL;
}