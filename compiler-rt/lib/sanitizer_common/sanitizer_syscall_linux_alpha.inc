//===-- sanitizer_syscall_linux_alpha.inc ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of internal_syscall and internal_iserror for Linux/Alpha.
//
// Alpha syscall ABI: number in $v0 ($0), args in $a0-$a5 ($16-$21), return
// value in $v0 ($0), error flag in $a3 ($19) -- 0 on success, 1 on error.
// On error, $v0 holds the positive errno.  Normalize to the standard -errno
// convention on return so that internal_iserror can use the generic test.
//
// For syscalls with 4-6 arguments $a3 ($19) serves dual purpose: it carries
// the fourth argument on entry and the error flag on return, so those variants
// use a read-write ("+r") constraint for that register.
//
//===----------------------------------------------------------------------===//

#define SYSCALL(name) __NR_##name

static uptr __internal_syscall(u64 nr) {
  register u64 v0 asm("$0") = nr;
  register u64 a3 asm("$19");
  asm volatile("callsys"
               : "+r"(v0), "=r"(a3)
               :
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall0(n) (__internal_syscall)(n)

static uptr __internal_syscall(u64 nr, u64 arg1) {
  register u64 v0 asm("$0") = nr;
  register u64 a0 asm("$16") = arg1;
  register u64 a3 asm("$19");
  asm volatile("callsys"
               : "+r"(v0), "=r"(a3)
               : "r"(a0)
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall1(n, a1) (__internal_syscall)(n, (u64)(a1))

static uptr __internal_syscall(u64 nr, u64 arg1, long arg2) {
  register u64 v0 asm("$0") = nr;
  register u64 a0 asm("$16") = arg1;
  register u64 a1 asm("$17") = arg2;
  register u64 a3 asm("$19");
  asm volatile("callsys"
               : "+r"(v0), "=r"(a3)
               : "r"(a0), "r"(a1)
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall2(n, a1, a2) \
  (__internal_syscall)(n, (u64)(a1), (long)(a2))

static uptr __internal_syscall(u64 nr, u64 arg1, long arg2, long arg3) {
  register u64 v0 asm("$0") = nr;
  register u64 a0 asm("$16") = arg1;
  register u64 a1 asm("$17") = arg2;
  register u64 a2 asm("$18") = arg3;
  register u64 a3 asm("$19");
  asm volatile("callsys"
               : "+r"(v0), "=r"(a3)
               : "r"(a0), "r"(a1), "r"(a2)
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall3(n, a1, a2, a3) \
  (__internal_syscall)(n, (u64)(a1), (long)(a2), (long)(a3))

static uptr __internal_syscall(u64 nr, u64 arg1, long arg2, long arg3,
                               u64 arg4) {
  register u64 v0 asm("$0") = nr;
  register u64 a0 asm("$16") = arg1;
  register u64 a1 asm("$17") = arg2;
  register u64 a2 asm("$18") = arg3;
  register u64 a3 asm("$19") = arg4;
  asm volatile("callsys"
               : "+r"(v0), "+r"(a3)
               : "r"(a0), "r"(a1), "r"(a2)
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall4(n, a1, a2, a3, a4) \
  (__internal_syscall)(n, (u64)(a1), (long)(a2), (long)(a3), (long)(a4))

static uptr __internal_syscall(u64 nr, u64 arg1, long arg2, long arg3, u64 arg4,
                               long arg5) {
  register u64 v0 asm("$0") = nr;
  register u64 a0 asm("$16") = arg1;
  register u64 a1 asm("$17") = arg2;
  register u64 a2 asm("$18") = arg3;
  register u64 a3 asm("$19") = arg4;
  register u64 a4 asm("$20") = arg5;
  asm volatile("callsys"
               : "+r"(v0), "+r"(a3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a4)
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall5(n, a1, a2, a3, a4, a5)                       \
  (__internal_syscall)(n, (u64)(a1), (long)(a2), (long)(a3), (long)(a4), \
                       (u64)(a5))

static uptr __internal_syscall(u64 nr, u64 arg1, long arg2, long arg3, u64 arg4,
                               long arg5, long arg6) {
  register u64 v0 asm("$0") = nr;
  register u64 a0 asm("$16") = arg1;
  register u64 a1 asm("$17") = arg2;
  register u64 a2 asm("$18") = arg3;
  register u64 a3 asm("$19") = arg4;
  register u64 a4 asm("$20") = arg5;
  register u64 a5 asm("$21") = arg6;
  asm volatile("callsys"
               : "+r"(v0), "+r"(a3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a4), "r"(a5)
               : "memory", "$1", "$22", "$23", "$24", "$25", "$28");
  return a3 ? -v0 : v0;
}
#define __internal_syscall6(n, a1, a2, a3, a4, a5, a6)                   \
  (__internal_syscall)(n, (u64)(a1), (long)(a2), (long)(a3), (long)(a4), \
                       (u64)(a5), (long)(a6))

#define __SYSCALL_NARGS_X(a1, a2, a3, a4, a5, a6, a7, a8, n, ...) n
#define __SYSCALL_NARGS(...) \
  __SYSCALL_NARGS_X(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0, )
#define __SYSCALL_CONCAT_X(a, b) a##b
#define __SYSCALL_CONCAT(a, b) __SYSCALL_CONCAT_X(a, b)
#define __SYSCALL_DISP(b, ...) \
  __SYSCALL_CONCAT(b, __SYSCALL_NARGS(__VA_ARGS__))(__VA_ARGS__)

#define internal_syscall(...) __SYSCALL_DISP(__internal_syscall, __VA_ARGS__)

bool internal_iserror(uptr retval, int* rverrno) {
  if (retval >= (uptr)-4095) {
    if (rverrno)
      *rverrno = -retval;
    return true;
  }
  return false;
}
