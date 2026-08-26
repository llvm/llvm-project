//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Freestanding code coverage extraction support for unit tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_COVERAGE_H
#define LLVM_LIBC_TEST_UNITTEST_COVERAGE_H

#include "src/__support/macros/properties/os.h"

#if defined(LIBC_TARGET_OS_IS_LINUX)

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_mman_macros.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getpid.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/munmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/write.h"
#include "src/__support/integer_to_string.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include <sys/syscall.h>

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

using LIBC_NAMESPACE::cpp::string_view;

/// Minimal fixed-size stack buffer for constructing file paths without dynamic
/// memory allocation.
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
  LIBC_NAMESPACE::linux_syscalls::write(2, msg.data(), msg.size());
}

} // anonymous namespace

/// Writes raw coverage profile data to disk using direct Linux syscalls.
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
  pid_t pid = LIBC_NAMESPACE::linux_syscalls::getpid();
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

#endif // LIBC_TARGET_OS_IS_LINUX

#endif // LLVM_LIBC_TEST_UNITTEST_COVERAGE_H
