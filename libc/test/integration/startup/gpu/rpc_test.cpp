//===-- Loader test to check the RPC interface with the loader ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "test/IntegrationTest/test.h"

using namespace __llvm_libc;

static void test_add_simple() {
  uint32_t num_additions =
      10 + 10 * gpu::get_thread_id() + 10 * gpu::get_block_id();
  uint64_t cnt = 0;
  for (uint32_t i = 0; i < num_additions; ++i) {
    rpc::Client::Port port = rpc::client.open<RPC_TEST_INCREMENT>();
    port.send_and_recv(
        [=](rpc::Buffer *buffer) {
          reinterpret_cast<uint64_t *>(buffer->data)[0] = cnt;
        },
        [&](rpc::Buffer *buffer) {
          cnt = reinterpret_cast<uint64_t *>(buffer->data)[0];
        });
    port.close();
  }
  ASSERT_TRUE(cnt == num_additions && "Incorrect sum");
}

// Test to ensure that the RPC mechanism doesn't hang on divergence.
static void test_noop(uint8_t data) {
  rpc::Client::Port port = rpc::client.open<RPC_NOOP>();
  port.send([=](rpc::Buffer *buffer) { buffer->data[0] = data; });
  port.close();
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_add_simple();

  if (gpu::get_thread_id() % 2)
    test_noop(1);
  else
    test_noop(2);

  return 0;
}
