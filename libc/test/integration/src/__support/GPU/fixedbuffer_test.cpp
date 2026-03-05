//===-- Integration test for the lock-free buffer -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/fixedbuffer.h"
#include "src/__support/GPU/utils.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

static FixedBuffer<uint32_t, 2048> global_buffer;

void run() {
  ASSERT_EQ(gpu::get_num_blocks() * gpu::get_num_threads(), 512);

  uint32_t id = static_cast<uint32_t>(gpu::get_thread_id()) + 1;
  uint32_t val;

  auto valid = [](uint32_t v, uint32_t n) {
    return (v >= 1 && v <= n) || v == UINT32_MAX;
  };

  for (int i = 0; i < 256; ++i) {
    EXPECT_TRUE(global_buffer.push(id));
    EXPECT_TRUE(global_buffer.pop(val));
    ASSERT_TRUE(valid(val, uint32_t(gpu::get_num_threads())));
  }

  EXPECT_TRUE(global_buffer.push(id));
  EXPECT_TRUE(global_buffer.push(id));
  EXPECT_TRUE(global_buffer.pop(val));
  ASSERT_TRUE(valid(val, uint32_t(gpu::get_num_threads())));

  while (!global_buffer.push(UINT32_MAX))
    ;
}

TEST_MAIN(int, char **, char **) {
  run();

  return 0;
}
