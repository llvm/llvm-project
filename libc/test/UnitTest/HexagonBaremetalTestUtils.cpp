//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains test utilities for bare-metal Hexagon hermetic tests.
///
/// llvm-libc's bare-metal port defers all OS interaction to a small set of
/// "vendor-provided" hook symbols (see
/// libc/src/__support/OSUtil/baremetal/io.h and the bare-metal time sources).
/// This file implements those hooks for the Hexagon simulator using Angel-style
/// semihosting, i.e. a `trap0(#0)` with:
///   R0 = system call code (also the return value)
///   R1 = pointer to an argument block (also errno on failure)
///   R2 = exit status (for SYS_EXIT only)
/// The read/write calls return the number of bytes NOT transferred.
///
/// It is compiled into HermeticTestUtils so hermetic tests can run under
/// the simulator. It is purely test-support code and is not intended for use as
/// general-purpose semihosting support.
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_timespec.h"

namespace {

enum HexagonSyscall {
  SYS_WRITE = 5,
  SYS_READ = 6,
  SYS_TIME = 0x11,
  SYS_EXIT = 24,
};

// Software interrupt used to enter the simulator monitor.
#define HEXAGON_SWI "trap0(#0)"

// Issue a semihosting call: R0 = code, R1 = pointer to argument block.
// Returns the value the simulator places in R0.
int hexagon_semihost(int code, size_t *args) {
  register uintptr_t r0 __asm__("r0") = static_cast<uintptr_t>(code);
  register uintptr_t r1 __asm__("r1") = reinterpret_cast<uintptr_t>(args);
  __asm__ __volatile__(HEXAGON_SWI : "=r"(r0), "=r"(r1) : "r"(r0), "r"(r1));
  return static_cast<int>(r0);
}

} // namespace

extern "C" {

// The cookie type/objects only need to exist; their contents encode the file
// descriptor used by the simulator-backed implementation.
struct __llvm_libc_stdio_cookie {
  int fd;
};

__llvm_libc_stdio_cookie __llvm_libc_stdin_cookie = {0};
__llvm_libc_stdio_cookie __llvm_libc_stdout_cookie = {1};
__llvm_libc_stdio_cookie __llvm_libc_stderr_cookie = {2};

ssize_t __llvm_libc_stdio_write(void *cookie, const char *buf, size_t size) {
  int fd = cookie ? static_cast<__llvm_libc_stdio_cookie *>(cookie)->fd : 1;
  size_t args[] = {
      static_cast<size_t>(fd),
      reinterpret_cast<size_t>(buf),
      size,
  };
  int not_written = hexagon_semihost(SYS_WRITE, args);
  if (not_written < 0)
    return -1;
  // The simulator returns the number of bytes NOT written.
  return static_cast<ssize_t>(size) - not_written;
}

ssize_t __llvm_libc_stdio_read(void *cookie, char *buf, size_t size) {
  int fd = cookie ? static_cast<__llvm_libc_stdio_cookie *>(cookie)->fd : 0;
  size_t args[] = {
      static_cast<size_t>(fd),
      reinterpret_cast<size_t>(buf),
      size,
  };
  int not_read = hexagon_semihost(SYS_READ, args);
  if (not_read < 0)
    return -1;
  // The simulator returns the number of bytes NOT read.
  return static_cast<ssize_t>(size) - not_read;
}

[[noreturn]] void __llvm_libc_exit(int status) {
  // The simulator reads the exit status from R2 and the syscall code from R0.
  register uintptr_t r2 __asm__("r2") = static_cast<uintptr_t>(status);
  register uintptr_t r0 __asm__("r0") = SYS_EXIT;
  __asm__ __volatile__(HEXAGON_SWI : : "r"(r0), "r"(r2) : "memory");
  __builtin_unreachable();
}

// The bare-metal config builds with LIBC_ERRNO_MODE_EXTERNAL, so the vendor
// must provide storage for errno via this hook.
int *__llvm_libc_errno() {
  static int errno_storage;
  return &errno_storage;
}

// Vendor hook used by the bare-metal `timespec_get()` implementation. The
// Hexagon simulator's SYS_TIME call returns wall-clock time in whole seconds
// (R0 = code on entry, R0 = seconds on return).
bool __llvm_libc_timespec_get_utc(struct timespec *ts) {
  register uintptr_t r0 __asm__("r0") = SYS_TIME;
  __asm__ __volatile__(HEXAGON_SWI : "=r"(r0) : "r"(r0));
  ts->tv_sec = static_cast<time_t>(r0);
  ts->tv_nsec = 0;
  return true;
}

} // extern "C"
