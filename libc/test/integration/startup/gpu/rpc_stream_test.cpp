//===-- Loader test to check the RPC streaming interface with the loader --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/integer_to_string.h"
#include "src/string/memory_utils/memcmp_implementations.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include "src/string/string_utils.h"
#include "test/IntegrationTest/test.h"

extern "C" void *malloc(uint64_t);
extern "C" void free(void *);

using namespace __llvm_libc;

static void test_stream() {
  const char str[] = "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy";
  uint64_t send_size = sizeof(str);
  void *send_ptr = malloc(send_size);
  void *recv_ptr;
  uint64_t recv_size;

  inline_memcpy(send_ptr, str, send_size);
  ASSERT_TRUE(inline_memcmp(send_ptr, str, send_size) == 0 && "Data mismatch");
  rpc::Client::Port port = rpc::client.open<rpc::TEST_STREAM>();
  port.send_n(send_ptr, send_size);
  port.recv_n(&recv_ptr, &recv_size,
              [](uint64_t size) { return malloc(size); });
  port.close();
  ASSERT_TRUE(inline_memcmp(recv_ptr, str, recv_size) == 0 && "Data mismatch");
  ASSERT_TRUE(recv_size == send_size && "Data size mismatch");

  free(send_ptr);
  free(recv_ptr);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_stream();

  return 0;
}
