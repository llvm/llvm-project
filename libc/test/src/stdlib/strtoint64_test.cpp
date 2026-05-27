//===-- Unittests for strtoint64 ------------------------------------------===//
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

int64_t strtoint64(const char *__restrict str, char **__restrict str_end,
                   int base) {
  return internal::str_to_helper<int64_t>(str, str_end, base);
}

uint64_t strtouint64(const char *__restrict str, char **__restrict str_end,
                     int base) {
  return internal::str_to_helper<uint64_t>(str, str_end, base);
}
} // namespace LIBC_NAMESPACE_DECL

STRTOL_TEST(Strtoint64, LIBC_NAMESPACE::strtoint64)
STRTOL_TEST(Strtouint64, LIBC_NAMESPACE::strtouint64)
