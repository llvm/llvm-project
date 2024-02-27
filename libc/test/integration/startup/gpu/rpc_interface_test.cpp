//===-- Loader test to check the RPC interface with the loader ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-libc-types/test_rpc_opcodes_t.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

// Test to ensure that we can use aribtrary combinations of sends and recieves
// as long as they are mirrored.
static void test_interface(bool end_with_send) {
  uint64_t cnt = 0;
  rpc::Client::Port port = rpc::client.open<RPC_TEST_INTERFACE>();
  port.send([&](rpc::Buffer *buffer) { buffer->data[0] = end_with_send; });
  port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  if (end_with_send)
    port.send([&](rpc::Buffer *buffer) { buffer->data[0] = cnt = cnt + 1; });
  else
    port.recv([&](rpc::Buffer *buffer) { cnt = buffer->data[0]; });
  port.close();

  ASSERT_TRUE(cnt == 9 && "Invalid number of increments");
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_interface(true);
  test_interface(false);

  return 0;
}
