//===-- Loader test to check the external RPC interface with the loader ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gpu/rpc.h>

#include "src/gpu/rpc_close_port.h"
#include "src/gpu/rpc_open_port.h"
#include "src/gpu/rpc_recv_n.h"
#include "src/gpu/rpc_send_n.h"

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

static void test_interface() {
  uint32_t num_additions =
      10 + 10 * gpu::get_thread_id() + 10 * gpu::get_block_id();
  uint64_t cnt = 0;
  for (uint32_t i = 0; i < num_additions; ++i) {
    size_t size = sizeof(uint64_t);
    rpc_port_t port = LIBC_NAMESPACE::rpc_open_port(RPC_TEST_EXTERNAL);
    LIBC_NAMESPACE::rpc_send_n(&port, &cnt, size);
    LIBC_NAMESPACE::rpc_recv_n(&port, &cnt, &size);
    LIBC_NAMESPACE::rpc_close_port(&port);
    ASSERT_TRUE(size == sizeof(uint64_t));
  }

  ASSERT_TRUE(cnt == num_additions && "Invalid number of increments");
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_interface();

  return 0;
}
