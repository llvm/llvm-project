//===-- sanitizer_syscall_linux_s390.inc ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of internal_syscall and internal_iserror for Linux/systemz.
//
//===----------------------------------------------------------------------===//
#define SYSCALL(name) __NR_ ## name

static uptr internal_syscall(u64 nr) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  asm volatile("svc 0" 
               : "=r"(retval) 
               : "r"(r1)
               : "memory", "cc");
  return retval;
}

template <typename T1>
static uptr internal_syscall(u64 nr, T1 arg1) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  register u64 r2 asm("r2") = (u64)arg1;
  asm volatile("svc 0" 
               : "=r"(retval) 
               : "r"(r1), "r"(r2)
               : "memory", "cc");
  return retval;
}

template <typename T1, typename T2>
static uptr internal_syscall(u64 nr, T1 arg1, T2 arg2) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  register u64 r2 asm("r2") = (u64)arg1;
  register u64 r3 asm("r3") = (u64)arg2;
  asm volatile("svc 0"
               : "=r"(retval)
               : "r"(r1), "r"(r2), "r"(r3)
               : "memory", "cc");
  return retval;
}

template <typename T1, typename T2, typename T3>
static uptr internal_syscall(u64 nr, T1 arg1, T2 arg2, T3 arg3) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  register u64 r2 asm("r2") = (u64)arg1;
  register u64 r3 asm("r3") = (u64)arg2;
  register u64 r4 asm("r4") = (u64)arg3;
  asm volatile("svc 0"
               : "=r"(retval)
               : "r"(r1), "r"(r2), "r"(r3), "r"(r4)
               : "memory", "cc");
  return retval;
}

template <typename T1, typename T2, typename T3, typename T4>
static uptr internal_syscall(u64 nr, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  register u64 r2 asm("r2") = (u64)arg1;
  register u64 r3 asm("r3") = (u64)arg2;
  register u64 r4 asm("r4") = (u64)arg3;
  register u64 r5 asm("r5") = (u64)arg4;
  asm volatile("svc 0"
               : "=r"(retval)
               : "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5)
               : "memory", "cc");
  return retval;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
static uptr internal_syscall(u64 nr, T1 arg1, T2 arg2, T3 arg3, T4 arg4,
                             T5 arg5) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  register u64 r2 asm("r2") = (u64)arg1;
  register u64 r3 asm("r3") = (u64)arg2;
  register u64 r4 asm("r4") = (u64)arg3;
  register u64 r5 asm("r5") = (u64)arg4;
  register u64 r6 asm("r6") = (u64)arg5;
  asm volatile("svc 0"
               : "=r"(retval)
               : "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5), "r"(r6)
               : "memory", "cc");
  return retval;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6>
static uptr internal_syscall(u64 nr, T1 arg1, T2 arg2, T3 arg3, T4 arg4,
                             T5 arg5, T6 arg6) {
  register u64 retval __asm__("r2");
  register u64 r1 asm("r1") = nr;
  register u64 r2 asm("r2") = (u64)arg1;
  register u64 r3 asm("r3") = (u64)arg2;
  register u64 r4 asm("r4") = (u64)arg3;
  register u64 r5 asm("r5") = (u64)arg4;
  register u64 r6 asm("r6") = (u64)arg5;
  register u64 r7 asm("r7") = (u64)arg6;
  asm volatile("svc 0"
               : "=r"(retval)
               : "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5), "r"(r6), "r"(r7)
               : "memory", "cc");
  return retval;
}

bool internal_iserror(uptr retval, int *rverrno) {
  if (retval >= (uptr)-4095) {
    if (rverrno)
      *rverrno = -retval;
    return true;
  }
  return false;
}
