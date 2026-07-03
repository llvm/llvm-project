//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of strcpy_s.
///
//===----------------------------------------------------------------------===//

#include "src/string/strcpy_s.h"
#define __STDC_WANT_LIB_EXT1__ 1
#include "hdr/stdint_proxy.h"
#undef __STDC_WANT_LIB_EXT1__
#include "hdr/types/constraint_handler_t.h"
#include "hdr/types/errno_t.h"
#include "hdr/types/rsize_t.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/common.h"
#include "src/__support/constraint_handler.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE static cpp::expected<size_t, const char *>
validate(char *__restrict s1, rsize_t s1max, const char *__restrict s2) {
  if (s1 == nullptr)
    return cpp::unexpected("strcpy_s: s1 cannot be null");
  if (s2 == nullptr)
    return cpp::unexpected("strcpy_s: s2 cannot be null");
  if (s1max > RSIZE_MAX)
    return cpp::unexpected("strcpy_s: s1max cannot exceed RSIZE_MAX");
  if (s1max == 0)
    return cpp::unexpected("strcpy_s: s1max cannot be 0");

  // count can't be greater than s1max by definition, so no need to check for
  // this case.
  size_t count = internal::strnlen_s(s2, s1max);
  if (s1max == count)
    return cpp::unexpected("strcpy_s: s1max is too small for s2");

  // Check overlap using the full regions defined by the standard, including the
  // terminating null byte:
  //   destination: [s1, s1 + s1max)
  //   source:      [s2, s2 + count + 1)
  // Use s1max for the destination's length, not count + 1, because the
  // standard allows for overwriting the entire destination region, even if
  // s2's length is smaller than s1max.
  const uintptr_t s1_addr = reinterpret_cast<uintptr_t>(s1);
  const uintptr_t s2_addr = reinterpret_cast<uintptr_t>(s2);
  if (s1_addr < s2_addr ? s2_addr - s1_addr < s1max
                        : s1_addr - s2_addr < count + 1)
    return cpp::unexpected("strcpy_s: s1 and s2 cannot overlap");

  return count;
}

LLVM_LIBC_FUNCTION(errno_t, strcpy_s,
                   (char *__restrict s1, rsize_t s1max,
                    const char *__restrict s2)) {
  const auto count_expected = validate(s1, s1max, s2);

  if (!count_expected.has_value()) {
    if (s1 != nullptr && s1max > 0 && s1max <= RSIZE_MAX) {
      s1[0] = '\0';
    }
    libc_global_constraint_handler(count_expected.error(), 0,
                                   LIBC_ERRNO_T_ERROR_VALUE);
    return LIBC_ERRNO_T_ERROR_VALUE;
  }

  inline_memcpy(s1, s2, count_expected.value() + 1);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
