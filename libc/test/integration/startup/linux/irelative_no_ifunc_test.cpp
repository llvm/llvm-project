//===-- Implementation of apply_irelative_relocs (no ifunc) test ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "startup/linux/irelative.h"
#include "test/IntegrationTest/test.h"

TEST_MAIN() {
  ASSERT_EQ(reinterpret_cast<uintptr_t>(__rela_iplt_start),
            reinterpret_cast<uintptr_t>(__rela_iplt_end));
  return 0;
}
