//===-- Integration test for the lock-free stack --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/fixedstack.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

void single_thread() {
  // FIXME: The NVPTX backend cannot handle atomic CAS on a local address space.
#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  FixedStack<int, 16> local_stack;

  for (int i = 0; i < 16; ++i)
    EXPECT_TRUE(local_stack.push(i));
  ASSERT_TRUE(local_stack.full());

  int val;
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(local_stack.pop(val));
    EXPECT_EQ(val, 16 - 1 - i);
  }
  ASSERT_TRUE(local_stack.empty());
#endif
}

static FixedStack<uint32_t, 2048> global_stack;
void multiple_threads() {
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

// Once all the threads have finished executing check the final state of the
// stack. Destructors are always run with a single thread on the GPU.
[[gnu::destructor]] void check_stack() {
  ASSERT_FALSE(global_stack.empty());

  while (!global_stack.empty()) {
    uint32_t val;
    ASSERT_TRUE(global_stack.pop(val));
    ASSERT_TRUE(val < 64 || val == UINT32_MAX);
  }
}

TEST_MAIN(int argc, char **argv, char **envp) {
  single_thread();

  multiple_threads();

  return 0;
}
