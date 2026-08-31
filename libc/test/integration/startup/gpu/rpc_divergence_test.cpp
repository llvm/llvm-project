//===-- Loader test to check the RPC interface with the loader ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"

#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

static inline uint32_t entropy() {
  return (static_cast<uint32_t>(gpu::processor_clock()) ^
          (gpu::get_thread_id_x() * 0x632be59b) ^
          (gpu::get_block_id_x() * 0x85157af5)) *
         0x9e3779bb;
}

static inline uint32_t xorshift32(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state * 0x9e3779bb;
}

void increment(uint64_t cnt) {
  LIBC_NAMESPACE::rpc::Client::Port port =
      LIBC_NAMESPACE::rpc::client.open<LIBC_TEST_INCREMENT>();
  port.send_and_recv(
      [=](LIBC_NAMESPACE::rpc::Buffer *buffer, uint32_t) {
        reinterpret_cast<uint64_t *>(buffer->data)[0] = cnt;
      },
      [&](LIBC_NAMESPACE::rpc::Buffer *buffer, uint32_t) {
        ASSERT_TRUE(reinterpret_cast<uint64_t *>(buffer->data)[0] == cnt + 1);
      });
}

TEST_MAIN(int, char **, char **) {
  uint32_t state = entropy();

  // Force a highly divergent warp state while hammering the RPC interface.
  uint32_t iters = 128 + xorshift32(state) % 128;
  for (uint32_t i = 0; i < iters; ++i) {
    if (xorshift32(state) % 127 == 0) {
      volatile int x = 0;
      uint32_t delay = xorshift32(state) % 4096;
      for (uint32_t j = 0; j < delay; ++j)
        x++;
    }
    uint32_t roll = xorshift32(state);
    if (roll % 2 == 0) {
      uint32_t burst = roll % 64 == 0 ? 2 + xorshift32(state) % 7 : 1;
      for (uint32_t b = 0; b < burst; ++b)
        increment(xorshift32(state));
    }
  }

  return 0;
}
