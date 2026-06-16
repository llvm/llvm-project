//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of wcstoumax.
///
//===----------------------------------------------------------------------===//

#include "src/inttypes/wcstoumax.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/str_to_integer.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(uintmax_t, wcstoumax,
                   (const wchar_t *__restrict str, wchar_t **__restrict str_end,
                    int base)) {
  auto result = internal::strtointeger<uintmax_t>(str, base);
  if (result.has_error())
    libc_errno = result.error;

  if (str_end != nullptr && result.error != EINVAL)
    *str_end = const_cast<wchar_t *>(str + result.parsed_len);

  return result;
}

} // namespace LIBC_NAMESPACE_DECL
