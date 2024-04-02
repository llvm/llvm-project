//===-- RPC test to check args to printf ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include "src/__support/GPU/utils.h"
#include "src/gpu/rpc_fprintf.h"
#include "src/stdio/fopen.h"

using namespace LIBC_NAMESPACE;

FILE *file = LIBC_NAMESPACE::fopen("testdata/test_data.txt", "w");

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(file && "failed to open file");
  // Check basic printing.
  int written = 0;
  written = LIBC_NAMESPACE::rpc_fprintf(file, "A simple string\n", nullptr, 0);
  ASSERT_EQ(written, 16);

  const char *str = "A simple string\n";
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%s", &str, sizeof(void *));
  ASSERT_EQ(written, 16);

  // Check printing a different value with each thread.
  uint64_t thread_id = gpu::get_thread_id();
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%8ld\n", &thread_id,
                                        sizeof(thread_id));
  ASSERT_EQ(written, 9);

  struct {
    uint32_t x = 1;
    char c = 'c';
    double f = 1.0;
  } args1;
  written =
      LIBC_NAMESPACE::rpc_fprintf(file, "%d%c%.1f\n", &args1, sizeof(args1));
  ASSERT_EQ(written, 6);

  struct {
    uint32_t x = 1;
    const char *str = "A simple string\n";
  } args2;
  written =
      LIBC_NAMESPACE::rpc_fprintf(file, "%032b%s\n", &args2, sizeof(args2));
  ASSERT_EQ(written, 49);

  // Check that the server correctly handles divergent numbers of arguments.
  const char *format = gpu::get_thread_id() % 2 ? "%s" : "%20ld\n";
  written = LIBC_NAMESPACE::rpc_fprintf(file, format, &str, sizeof(void *));
  ASSERT_EQ(written, gpu::get_thread_id() % 2 ? 16 : 21);

  format = gpu::get_thread_id() % 2 ? "%s" : str;
  written = LIBC_NAMESPACE::rpc_fprintf(file, format, &str, sizeof(void *));
  ASSERT_EQ(written, 16);

  // Check that we handle null arguments correctly.
  struct {
    void *null = nullptr;
  } args3;
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%p", &args3, sizeof(args3));
  ASSERT_EQ(written, 9);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%s", &args3, sizeof(args3));
  ASSERT_EQ(written, 6);
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS

  // Check for extremely abused variable width arguments
  struct {
    uint32_t x = 1;
    uint32_t y = 2;
    double f = 1.0;
  } args4;
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%**d", &args4, sizeof(args4));
  ASSERT_EQ(written, 4);
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%**d%6d", &args4, sizeof(args4));
  ASSERT_EQ(written, 10);
  written = LIBC_NAMESPACE::rpc_fprintf(file, "%**.**f", &args4, sizeof(args4));
  ASSERT_EQ(written, 7);

  return 0;
}
