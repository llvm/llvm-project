//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Arm bare-metal hermetic test utilities.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_timespec.h"

// Redirect stdout/stderr, time and exit from libc tests to semihosting

namespace {
// Semihosting constants and semihosting call wrapper.
// https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst
constexpr uint32_t SYS_OPEN = 0x01;
constexpr uint32_t SYS_WRITE = 0x05;
constexpr uint32_t SYS_CLOCK = 0x10;
constexpr uint32_t SYS_TIME = 0x11;
constexpr uint32_t SYS_EXIT = 0x18;

constexpr uint32_t OPENMODE_W = 4;

constexpr uint32_t ADP_Stopped_ApplicationExit = 0x20026;
constexpr uint32_t ADP_Stopped_RunTimeErrorUnknown = 0x20023;

#if defined(__thumb__) // T32
#if defined(__ARM_ARCH_PROFILE) && __ARM_ARCH_PROFILE == 'M'
#define SEMIHOST_INSTRUCTION "bkpt #0xAB"
#else
#define SEMIHOST_INSTRUCTION "svc 0xab"
#endif
#else // A32
#define SEMIHOST_INSTRUCTION "svc 0x123456"
#endif

long semihosting_call(long val, const void *ptr) {
  register long v __asm__("r0") = val;
  register const void *p __asm__("r1") = ptr;
  __asm__ __volatile__(SEMIHOST_INSTRUCTION
                       : "+r"(v), "+r"(p)
                       :
                       : "memory", "cc");
  return v;
}
} // namespace

extern "C" {
struct __llvm_libc_stdio_cookie {
  int handle;
};

struct __llvm_libc_stdio_cookie __llvm_libc_stdout_cookie;
struct __llvm_libc_stdio_cookie __llvm_libc_stderr_cookie;

static void stdio_open(struct __llvm_libc_stdio_cookie *cookie, size_t mode) {
  const char std_stream_name[] = ":tt";
  size_t args[] = {
      reinterpret_cast<size_t>(std_stream_name),
      mode,
      sizeof(std_stream_name) - 1UL,
  };
  cookie->handle = semihosting_call(SYS_OPEN, args);
}

void _platform_init(void) {
  stdio_open(&__llvm_libc_stdout_cookie, OPENMODE_W);
  stdio_open(&__llvm_libc_stderr_cookie, OPENMODE_W);
}

void __llvm_libc_exit(int status) {
  uint32_t semihosting_status = (status == 0) ? ADP_Stopped_ApplicationExit
                                              : ADP_Stopped_RunTimeErrorUnknown;
  semihosting_call(SYS_EXIT,
                   reinterpret_cast<const void *>(semihosting_status));
  __builtin_unreachable(); // This semihosting call does not return.
}

ssize_t __llvm_libc_stdio_write(struct __llvm_libc_stdio_cookie *cookie,
                                const char *buf, size_t size) {
  size_t args[] = {
      static_cast<size_t>(cookie->handle),
      reinterpret_cast<size_t>(buf),
      size,
  };
  ssize_t retval = semihosting_call(SYS_WRITE, args);
  if (retval >= 0)
    retval = size - retval;
  return retval;
}

bool __llvm_libc_timespec_get_active(struct timespec *ts) {
  long retval = semihosting_call(SYS_CLOCK, 0);
  if (retval == -1)
    return false;

  // Semihosting uses centiseconds.
  ts->tv_sec = (retval / 100);
  ts->tv_nsec = (retval % 100) * (1'000'000'000 / 100);
  return true;
}

bool __llvm_libc_timespec_get_utc(struct timespec *ts) {
  long retval = semihosting_call(SYS_TIME, 0);

  // Semihosting uses seconds.
  ts->tv_sec = retval;
  ts->tv_nsec = 0;
  return true;
}
} // extern "C"
