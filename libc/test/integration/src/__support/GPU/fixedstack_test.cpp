//===-- Integration test for the lock-free stack --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/fixedstack.h"
#include "src/__support/GPU/utils.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

static FixedStack<uint32_t, 2048> global_stack;

void run() {
  // We need enough space in the stack as threads in flight can temporarily
  // consume memory before they finish comitting it back to the stack.
  ASSERT_EQ(gpu::get_num_blocks() * gpu::get_num_threads(), 512);

  uint32_t val;
  uint32_t num_threads = static_cast<uint32_t>(gpu::get_num_threads());
  for (int i = 0; i < 256; ++i) {
    EXPECT_TRUE(global_stack.push(UINT32_MAX))
    EXPECT_TRUE(global_stack.pop(val))
    ASSERT_TRUE(val < num_threads || val == UINT32_MAX);
  }

  EXPECT_TRUE(global_stack.push(static_cast<uint32_t>(gpu::get_thread_id())));
  EXPECT_TRUE(global_stack.push(static_cast<uint32_t>(gpu::get_thread_id())));
  EXPECT_TRUE(global_stack.pop(val));
  ASSERT_TRUE(val < num_threads || val == UINT32_MAX);

  // Fill the rest of the stack with the default value.
  while (!global_stack.push(UINT32_MAX))
    ;
}

TEST_MAIN(int argc, char **argv, char **envp) {
  run();

  return 0;
}
