//===-- Main function for implementation of base class for libc unittests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"
#include "src/__support/CPP/string_view.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::testing::TestOptions;

namespace {

// A poor-man's getopt_long.
// Run unit tests with --gtest_color=no to disable printing colors, or
// --gtest_print_time to print timings in milliseconds only (as GTest does, so
// external tools such as Android's atest may expect that format to parse the
// output). Other command line flags starting with --gtest_ are ignored.
// Otherwise, the last command line arg is used as a test filter, if command
// line args are specified.
TestOptions parseOptions(int argc, char **argv) {
  TestOptions Options;

  for (int i = 1; i < argc; ++i) {
    string_view arg{argv[i]};

    if (arg == "--gtest_color=no")
      Options.PrintColor = false;
    else if (arg == "--gtest_print_time")
      Options.TimeInMs = true;
    // Ignore other unsupported gtest specific flags.
    else if (arg.starts_with("--gtest_"))
      continue;
    else
      Options.TestFilter = argv[i];
  }

  return Options;
}

} // anonymous namespace

#if defined(__linux__)
#include "src/__support/OSUtil/syscall.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>

extern "C" {
__attribute__((weak)) uint64_t __llvm_profile_get_size_for_buffer(void);
__attribute__((weak)) int __llvm_profile_write_buffer(char *Buffer);
__attribute__((weak)) void __llvm_profile_set_filename(const char *FilenamePat);

// Override compiler-rt's weak filename symbol. This redirects the default
// filename to /dev/null to silence the default dumper by default.
char __llvm_profile_filename[] = "/dev/null";
}

namespace {
void write_raw_profile() {
  if (!__llvm_profile_get_size_for_buffer || !__llvm_profile_write_buffer)
    return;

  uint64_t required_size = __llvm_profile_get_size_for_buffer();
  if (required_size == 0)
    return;

  long mmap_ret = LIBC_NAMESPACE::syscall_impl<long>(
      SYS_mmap, nullptr, required_size, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mmap_ret < 0 && mmap_ret > -4096)
    return;
  char *profile_buffer = reinterpret_cast<char *>(mmap_ret);

  if (__llvm_profile_write_buffer(profile_buffer) != 0) {
    LIBC_NAMESPACE::syscall_impl<long>(SYS_munmap, profile_buffer, required_size);
    return;
  }

  // Create a minimal filename: libc_cov_<pid>.profraw
  long pid = LIBC_NAMESPACE::syscall_impl<long>(SYS_getpid);
  if (pid <= 0) pid = 1;

  char filename[64] = "libc_cov_";
  int idx = 9;
  
  long temp = pid;
  char digits[32];
  int d_len = 0;
  while (temp > 0) { digits[d_len++] = static_cast<char>('0' + (temp % 10)); temp /= 10; }
  if (d_len == 0) digits[d_len++] = '0';
  for (int i = d_len - 1; i >= 0; --i) filename[idx++] = digits[i];
  
  const char *suffix = ".profraw";
  for (int i = 0; suffix[i] != '\0'; ++i) filename[idx++] = suffix[i];
  filename[idx] = '\0';

  long fd = LIBC_NAMESPACE::syscall_impl<long>(
      SYS_openat, AT_FDCWD, filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      
  if (fd >= 0) {
    uint64_t bytes_written = 0;
    while (bytes_written < required_size) {
      long ret = LIBC_NAMESPACE::syscall_impl<long>(
          SYS_write, fd, profile_buffer + bytes_written, required_size - bytes_written);
      if (ret <= 0 && ret != -EINTR) break;
      if (ret > 0) bytes_written += ret;
    }
    LIBC_NAMESPACE::syscall_impl<long>(SYS_close, fd);
  }

  LIBC_NAMESPACE::syscall_impl<long>(SYS_munmap, profile_buffer, required_size);

  // Clear the filename pattern to prevent compiler-rt from writing at exit.
  if (__llvm_profile_set_filename)
    __llvm_profile_set_filename("/dev/null");
}
} // anonymous namespace
#else
namespace {
void write_raw_profile() {}
} // anonymous namespace
#endif

#if __STDC_HOSTED__
#define TEST_MAIN int main
#else
#define TEST_MAIN extern "C" int main
#endif

TEST_MAIN(int argc, char **argv, char **envp) {
  LIBC_NAMESPACE::testing::argc = argc;
  LIBC_NAMESPACE::testing::argv = argv;
  LIBC_NAMESPACE::testing::envp = envp;

  int result =
      LIBC_NAMESPACE::testing::Test::runTests(parseOptions(argc, argv));
  write_raw_profile();
  return result;
}
