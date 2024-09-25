//===-- RPC test to check args to printf ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include "src/__support/GPU/utils.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fprintf.h"

using namespace LIBC_NAMESPACE;

FILE *file = LIBC_NAMESPACE::fopen("testdata/test_data.txt", "w");

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(file && "failed to open file");
  // Check basic printing.
  int written = 0;
  written = LIBC_NAMESPACE::fprintf(file, "A simple string\n");
  ASSERT_EQ(written, 16);

  const char *str = "A simple string\n";
  written = LIBC_NAMESPACE::fprintf(file, "%s", str);
  ASSERT_EQ(written, 16);

  // Check printing a different value with each thread.
  uint64_t thread_id = gpu::get_thread_id();
  written = LIBC_NAMESPACE::fprintf(file, "%8ld\n", thread_id);
  ASSERT_EQ(written, 9);

  written = LIBC_NAMESPACE::fprintf(file, "%d%c%.1f\n", 1, 'c', 1.0);
  ASSERT_EQ(written, 6);

  written = LIBC_NAMESPACE::fprintf(file, "%032b%s\n", 1, "A simple string\n");
  ASSERT_EQ(written, 49);

  // Check that the server correctly handles divergent numbers of arguments.
  const char *format = gpu::get_thread_id() % 2 ? "%s" : "%20ld\n";
  written = LIBC_NAMESPACE::fprintf(file, format, str);
  ASSERT_EQ(written, gpu::get_thread_id() % 2 ? 16 : 21);

  format = gpu::get_thread_id() % 2 ? "%s" : str;
  written = LIBC_NAMESPACE::fprintf(file, format, str);
  ASSERT_EQ(written, 16);

  // Check that we handle null arguments correctly.
  written = LIBC_NAMESPACE::fprintf(file, "%p", nullptr);
  ASSERT_EQ(written, 9);

#ifndef LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
  written = LIBC_NAMESPACE::fprintf(file, "%s", nullptr);
  ASSERT_EQ(written, 6);
#endif // LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS

  // Check for extremely abused variable width arguments
  written = LIBC_NAMESPACE::fprintf(file, "%**d", 1, 2, 1.0);
  ASSERT_EQ(written, 4);
  written = LIBC_NAMESPACE::fprintf(file, "%**d%6d", 1, 2, 1.0);
  ASSERT_EQ(written, 10);
  written = LIBC_NAMESPACE::fprintf(file, "%**.**f", 1, 2, 1.0);
  ASSERT_EQ(written, 7);

  return 0;
}
