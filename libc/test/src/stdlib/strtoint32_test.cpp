//===-- Unittests for strtoint32 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "src/__support/str_to_integer.h"
#include "src/errno/libc_errno.h"

#include "StrtolTest.h"
#include "test/UnitTest/Test.h"

namespace __llvm_libc {

int32_t strtoint32(const char *__restrict str, char **__restrict str_end,
                   int base) {
  auto result = internal::strtointeger<int32_t>(str, base);
  if (result.has_error())
    libc_errno = result.error;

  if (str_end != nullptr)
    *str_end = const_cast<char *>(str + result.parsed_len);

  return result;
}

uint32_t strtouint32(const char *__restrict str, char **__restrict str_end,
                     int base) {
  auto result = internal::strtointeger<uint32_t>(str, base);
  if (result.has_error())
    libc_errno = result.error;

  if (str_end != nullptr)
    *str_end = const_cast<char *>(str + result.parsed_len);

  return result;
}
} // namespace __llvm_libc

STRTOL_TEST(Strtoint32, __llvm_libc::strtoint32)
STRTOL_TEST(Strtouint32, __llvm_libc::strtouint32)
