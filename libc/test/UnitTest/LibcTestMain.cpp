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

#include "src/__support/macros/properties/os.h"
#if defined(LIBC_TARGET_OS_IS_LINUX)
#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_mman_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/munmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/write.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/integer_to_string.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include <sys/syscall.h>

//===----------------------------------------------------------------------===//
// Freestanding Linux Code Coverage Profile Writer
//
// Freestanding (-nostdlib) libc binaries cannot link standard compiler-rt
// file I/O (fopen/fwrite). Here we override compiler-rt's default filename
// to "/dev/null" and invoke write_raw_profile() directly before main()
// returns (and within death test subprocesses in ExecuteFunctionUnix.cpp)
// to dump raw coverage counters (libc_cov_<pid>.profraw) using direct Linux
// system calls (SYS_mmap, SYS_openat, SYS_write, SYS_close, SYS_munmap).
//===----------------------------------------------------------------------===//
extern "C" {
__attribute__((weak)) uint64_t __llvm_profile_get_size_for_buffer();
__attribute__((weak)) int __llvm_profile_write_buffer(char *buffer);
__attribute__((weak)) void
__llvm_profile_set_filename(const char *filename_pat);

// Override compiler-rt's weak filename symbol. This redirects the default
// filename to /dev/null to silence the default dumper by default.
__attribute__((weak)) char __llvm_profile_filename[] = "/dev/null";
}

namespace {
struct FixedSizeBuffer {
  char data[64];
  size_t idx = 0;

  FixedSizeBuffer() { data[0] = '\0'; }

  bool append(string_view str) {
    size_t len = str.size();
    if (idx + len >= sizeof(data))
      return false;
    LIBC_NAMESPACE::inline_memcpy(data + idx, str.data(), len);
    idx += len;
    data[idx] = '\0';
    return true;
  }

  template <size_t N> bool append(const char (&str)[N]) {
    size_t len = N - 1;
    if (idx + len >= sizeof(data))
      return false;
    LIBC_NAMESPACE::inline_memcpy(data + idx, str, len);
    idx += len;
    data[idx] = '\0';
    return true;
  }
};

LIBC_INLINE void report_error(string_view msg) {
  LIBC_NAMESPACE::syscall_impl<long>(SYS_write, 2, msg.data(), msg.size());
}
} // anonymous namespace

extern "C" void write_raw_profile() {
  if (!__llvm_profile_get_size_for_buffer || !__llvm_profile_write_buffer)
    return;

  size_t required_size =
      static_cast<size_t>(__llvm_profile_get_size_for_buffer());
  if (required_size == 0)
    return;

  auto mmap_or_error = LIBC_NAMESPACE::linux_syscalls::mmap(
      nullptr, required_size, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!mmap_or_error)
    return report_error("error: libc coverage failed to mmap buffer\n");
  char *profile_buffer = static_cast<char *>(mmap_or_error.value());

  if (__llvm_profile_write_buffer(profile_buffer) != 0) {
    LIBC_NAMESPACE::linux_syscalls::munmap(profile_buffer, required_size);
    return report_error(
        "error: libc coverage failed to write profile buffer\n");
  }

  // Create a minimal filename: libc_cov_<pid>.profraw
  long pid = LIBC_NAMESPACE::syscall_impl<long>(SYS_getpid);
  if (pid <= 0)
    pid = 1;

  FixedSizeBuffer filename;
  char pid_buf[LIBC_NAMESPACE::IntegerToString<long>::buffer_size()];
  auto pid_str = LIBC_NAMESPACE::IntegerToString<long>::format_to(pid_buf, pid);
  if (!pid_str || !filename.append("libc_cov_") || !filename.append(*pid_str) ||
      !filename.append(".profraw")) {
    LIBC_NAMESPACE::linux_syscalls::munmap(profile_buffer, required_size);
    return report_error("error: libc coverage filename buffer overflow\n");
  }

  auto fd_or_error = LIBC_NAMESPACE::linux_syscalls::open(
      filename.data, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (!fd_or_error) {
    LIBC_NAMESPACE::linux_syscalls::munmap(profile_buffer, required_size);
    return report_error("error: libc coverage failed to open output file\n");
  }
  int fd = fd_or_error.value();

  size_t bytes_written = 0;
  bool write_error_occurred = false;
  while (bytes_written < required_size) {
    auto write_or_error = LIBC_NAMESPACE::linux_syscalls::write(
        fd, profile_buffer + bytes_written, required_size - bytes_written);
    if (!write_or_error) {
      if (write_or_error.error() == EINTR)
        continue;
      write_error_occurred = true;
      break;
    }
    ssize_t ret = write_or_error.value();
    if (ret == 0) {
      write_error_occurred = true;
      break;
    }
    bytes_written += ret;
  }

  LIBC_NAMESPACE::linux_syscalls::close(fd);
  LIBC_NAMESPACE::linux_syscalls::munmap(profile_buffer, required_size);

  if (write_error_occurred || bytes_written < required_size)
    return report_error(
        "error: libc coverage failed to write all data to file\n");

  // Clear the filename pattern to prevent compiler-rt from writing at exit.
  if (__llvm_profile_set_filename)
    __llvm_profile_set_filename("/dev/null");
}
#else
extern "C" void write_raw_profile() {}
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
