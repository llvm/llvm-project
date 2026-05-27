//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared implementation of strto* and wcsto* endpoints.
///
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/str_to_integer.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Shared implementation of strto* and wcsto* endpoints. Invokes
// strtointeger shared API and sets errno and str_end pointer according to the
// standard.
template <typename T, typename CharType>
LIBC_INLINE constexpr T str_to_helper(const CharType *__restrict str,
                                      CharType **__restrict str_end, int base) {
  auto result = strtointeger<T>(str, base);
  if (result.has_error())
    libc_errno = result.error;

  // It is unspecified whether str_end should be set to "str" if the base
  // is invalid, we explicitly avoid setting it for consistency with
  // other implementations.
  if (str_end != nullptr && result.error != EINVAL)
    *str_end = const_cast<CharType *>(str + result.parsed_len);

  return result.value;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
