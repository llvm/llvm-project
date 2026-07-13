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
/// compatible memcpy and memmove functions to be used when their assembly-
/// optimized counterparts can't.
///
//===----------------------------------------------------------------------===//

#include <stddef.h>

static void *__arm_sc_memcpy_fwd(void *dest, const void *src,
                                 size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  for (size_t i = 0; i < n; ++i)
    destp[i] = srcp[i];
  return dest;
}

static void *__arm_sc_memcpy_rev(void *dest, const void *src,
                                 size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  while (n > 0) {
    --n;
    destp[n] = srcp[n];
  }
  return dest;
}

extern void *__arm_sc_memcpy(void *__restrict dest, const void *__restrict src,
                             size_t n) __arm_streaming_compatible {
  return __arm_sc_memcpy_fwd(dest, src, n);
}

extern void *__arm_sc_memmove(void *dest, const void *src,
                              size_t n) __arm_streaming_compatible {
  unsigned char *destp = (unsigned char *)dest;
  const unsigned char *srcp = (const unsigned char *)src;

  if ((srcp > (destp + n)) || (destp > (srcp + n)))
    return __arm_sc_memcpy(dest, src, n);
  if (srcp > destp)
    return __arm_sc_memcpy_fwd(dest, src, n);
  return __arm_sc_memcpy_rev(dest, src, n);
}
