//===-- Implementation of lsearch -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/lsearch.h"
#include "src/__support/CPP/cstddef.h" // cpp::byte
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/memory_size.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void *, lsearch,
                   (const void *key, void *base, size_t *nmemb, size_t size,
                    int (*compar)(const void *, const void *))) {
  if (key == nullptr || base == nullptr || nmemb == nullptr ||
      compar == nullptr)
    return nullptr;

  size_t byte_len = 0;
  if (internal::mul_overflow(*nmemb, size, &byte_len))
    return nullptr;

  const cpp::byte *next = reinterpret_cast<const cpp::byte *>(base);
  const cpp::byte *end = next + byte_len;
  for (; next < end; next += size)
    if (compar(key, next) == 0)
      return const_cast<cpp::byte *>(next);

  *nmemb += 1;
  inline_memcpy(const_cast<cpp::byte *>(end), key, size);
  return const_cast<cpp::byte *>(end);
}

} // namespace LIBC_NAMESPACE_DECL
