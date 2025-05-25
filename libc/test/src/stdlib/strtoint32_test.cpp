//===-- Unittests for strtoint32 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "src/__support/macros/config.h"
#include "src/__support/str_to_integer.h"
#include "src/errno/libc_errno.h"

#include "StrtolTest.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

int32_t strtoint32(const char *__restrict str, char **__restrict str_end,
                   int base) {
  auto result = internal::strtointeger<int32_t>(str, base);
  if (result.has_error())
    LIBC_NAMESPACE::libc_errno = result.error;

  if (str_end != nullptr)
    *str_end = const_cast<char *>(str + result.parsed_len);

  return result;
}

uint32_t strtouint32(const char *__restrict str, char **__restrict str_end,
                     int base) {
  auto result = internal::strtointeger<uint32_t>(str, base);
  if (result.has_error())
    LIBC_NAMESPACE::libc_errno = result.error;

  if (str_end != nullptr)
    *str_end = const_cast<char *>(str + result.parsed_len);

  return result;
}
} // namespace LIBC_NAMESPACE_DECL

STRTOL_TEST(Strtoint32, LIBC_NAMESPACE::strtoint32)
STRTOL_TEST(Strtouint32, LIBC_NAMESPACE::strtouint32)
