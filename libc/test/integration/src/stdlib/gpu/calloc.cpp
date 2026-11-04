//===-- GPU calloc allocator compatibility test ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"

#ifdef LIBC_TARGET_ARCH_IS_NVPTX
extern "C" void free(void *);
#endif

TEST_MAIN(int, char **, char **) {
  for (int iteration = 0; iteration < 32; ++iteration) {
    auto *p = static_cast<unsigned char *>(LIBC_NAMESPACE::calloc(17, 4));
    ASSERT_NE(p, nullptr);
    for (int i = 0; i < 68; ++i)
      ASSERT_EQ(p[i], 0);
    p[0] = 1;
    p[67] = 2;
#ifdef LIBC_TARGET_ARCH_IS_NVPTX
    ::free(p);
#else
    LIBC_NAMESPACE::free(p);
#endif
  }

  return 0;
}
