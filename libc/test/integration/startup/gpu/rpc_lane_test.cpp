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

using namespace LIBC_NAMESPACE;

static void test_add() {
  uint64_t cnt = gpu::get_lane_id();
  LIBC_NAMESPACE::rpc::Client::Port port =
      LIBC_NAMESPACE::rpc::client.open<RPC_TEST_INCREMENT>();
  port.send_and_recv(
      [=](LIBC_NAMESPACE::rpc::Buffer *buffer, uint32_t) {
        reinterpret_cast<uint64_t *>(buffer->data)[0] = cnt;
      },
      [&](LIBC_NAMESPACE::rpc::Buffer *buffer, uint32_t) {
        cnt = reinterpret_cast<uint64_t *>(buffer->data)[0];
      });
  port.close();
  EXPECT_EQ(cnt, gpu::get_lane_id() + 1);
  EXPECT_EQ(gpu::get_thread_id(), gpu::get_lane_id());
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_add();

  return 0;
}
