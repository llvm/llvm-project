//===-- Unittests for strtoint32 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/str_to_util.h"

#include "StrtolTest.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

int32_t strtoint32(const char *__restrict str, char **__restrict str_end,
                   int base) {
  return internal::str_to_helper<int32_t>(str, str_end, base);
}

uint32_t strtouint32(const char *__restrict str, char **__restrict str_end,
                     int base) {
  return internal::str_to_helper<uint32_t>(str, str_end, base);
}
} // namespace LIBC_NAMESPACE_DECL

STRTOL_TEST(Strtoint32, LIBC_NAMESPACE::strtoint32)
STRTOL_TEST(Strtouint32, LIBC_NAMESPACE::strtouint32)
