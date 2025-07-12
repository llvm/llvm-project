//===-- Test for parallel GPU malloc interface ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include "src/__support/GPU/utils.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"

using namespace LIBC_NAMESPACE;

static inline void use(uint8_t *ptr, uint32_t size) {
  EXPECT_NE(ptr, nullptr);
  for (int i = 0; i < size; ++i)
    ptr[i] = uint8_t(i + gpu::get_thread_id());

  // Try to detect if some other thread manages to clobber our memory.
  for (int i = 0; i < size; ++i)
    EXPECT_EQ(ptr[i], uint8_t(i + gpu::get_thread_id()));
}

TEST_MAIN(int, char **, char **) {
  void *ptrs[256];
  for (int i = 0; i < 256; ++i)
    ptrs[i] = malloc(gpu::get_lane_id() % 2 ? 16 : 32);

  for (int i = 0; i < 256; ++i)
    use(reinterpret_cast<uint8_t *>(ptrs[i]), gpu::get_lane_id() % 2 ? 16 : 32);

  for (int i = 0; i < 256; ++i)
    free(ptrs[i]);
  return 0;
}
