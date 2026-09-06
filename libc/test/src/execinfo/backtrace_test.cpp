//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the backtrace calls
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/execinfo/backtrace.h"
#include "src/execinfo/backtrace_symbols_fd.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/fstat.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/Test.h"

namespace {

// Three frames deep, so the trace has something in it to count. They are
// kept from being folded into one another by -fno-optimize-sibling-calls.
[[gnu::noinline]] int innermost(void **buffer, int size) {
  return LIBC_NAMESPACE::backtrace(buffer, size);
}

[[gnu::noinline]] int middle(void **buffer, int size) {
  return innermost(buffer, size);
}

} // anonymous namespace

// A program with no unwinder to reach has no stack to walk, and says so by
// reporting no frames. That is what a test linked statically against a libc
// built without one gets, and it is an answer rather than a failure.
TEST(LlvmLibcBacktraceTest, WalksBackUpTheStack) {
  void *buffer[32] = {};
  int count = middle(buffer, 32);
  ASSERT_GE(count, 0);
  ASSERT_LE(count, 32);
  if (count == 0)
    return;

  // The three frames here at least, and every one of them an address.
  ASSERT_GE(count, 3);
  for (int i = 0; i < count; ++i)
    ASSERT_FALSE(buffer[i] == nullptr);
}

// A buffer smaller than the stack is filled and no more.
TEST(LlvmLibcBacktraceTest, StopsAtTheRoomItWasGiven) {
  void *buffer[2] = {};
  int count = middle(buffer, 2);
  ASSERT_GE(count, 0);
  ASSERT_LE(count, 2);

  ASSERT_EQ(LIBC_NAMESPACE::backtrace(buffer, 0), 0);
  ASSERT_EQ(LIBC_NAMESPACE::backtrace(nullptr, 32), 0);
}

// The fd form writes the names out without allocating anything. With no
// frames to name it writes nothing, which is still not an error.
TEST(LlvmLibcBacktraceTest, WritesTheNamesToADescriptor) {
  void *buffer[32] = {};
  int count = middle(buffer, 32);

  auto path = libc_make_test_file_path("backtrace.test");
  int fd = LIBC_NAMESPACE::open(path, O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
  ASSERT_GT(fd, 0);

  LIBC_NAMESPACE::backtrace_symbols_fd(buffer, count, fd);

  struct stat st;
  ASSERT_EQ(LIBC_NAMESPACE::fstat(fd, &st), 0);
  if (count > 0)
    ASSERT_GT(st.st_size, static_cast<off_t>(0));

  ASSERT_EQ(LIBC_NAMESPACE::close(fd), 0);
  ASSERT_EQ(LIBC_NAMESPACE::unlink(path), 0);
}
