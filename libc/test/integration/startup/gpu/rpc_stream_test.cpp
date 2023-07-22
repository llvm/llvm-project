//===-- Loader test to check the RPC streaming interface with the loader --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/integer_to_string.h"
#include "src/string/memory_utils/inline_memcmp.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"
#include "test/IntegrationTest/test.h"

extern "C" void *malloc(uint64_t);
extern "C" void free(void *);

using namespace __llvm_libc;

static void test_stream() {
  static const char str[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvwxy"
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
  rpc::Client::Port port = rpc::client.open<RPC_TEST_STREAM>();
  port.send_n(send_ptr, send_size);
  port.recv_n(&recv_ptr, &recv_size,
              [](uint64_t size) { return malloc(size); });
  port.close();
  ASSERT_TRUE(inline_memcmp(recv_ptr, str, recv_size) == 0 && "Data mismatch");
  ASSERT_TRUE(recv_size == send_size && "Data size mismatch");

  free(send_ptr);
  free(recv_ptr);
}

static void test_divergent() {
  static const uint8_t data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
      90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
      135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
      165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
      180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
      195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
      210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
      225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
      240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
      255,
  };

  uint8_t buffer[128] = {0};
  uint64_t offset =
      (gpu::get_thread_id() + gpu::get_num_threads() * gpu::get_block_id()) %
      128;
  void *recv_ptr;
  uint64_t recv_size;
  inline_memcpy(buffer, &data[offset], offset);
  ASSERT_TRUE(inline_memcmp(buffer, &data[offset], offset) == 0 &&
              "Data mismatch");
  rpc::Client::Port port = rpc::client.open<RPC_TEST_STREAM>();
  port.send_n(buffer, offset);
  inline_memset(buffer, offset, 0);
  port.recv_n(&recv_ptr, &recv_size, [&](uint64_t) { return buffer; });
  port.close();

  ASSERT_TRUE(inline_memcmp(recv_ptr, &data[offset], recv_size) == 0 &&
              "Data mismatch");
  ASSERT_TRUE(recv_size == offset && "Data size mismatch");
}

TEST_MAIN(int argc, char **argv, char **envp) {
  test_stream();
  test_divergent();

  return 0;
}
