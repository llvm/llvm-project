//===--------- inline implementation of arm syscalls --------------* C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_ARM_SYSCALL_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_ARM_SYSCALL_H

#include "src/__support/common.h"

#ifdef __thumb__
#error "The arm syscall implementation does not yet support thumb flavor."
#endif // __thumb__

#define REGISTER_DECL_0                                                        \
  register long r7 __asm__("r7") = number;                                     \
  register long r0 __asm__("r0");
#define REGISTER_DECL_1                                                        \
  register long r7 __asm__("r7") = number;                                     \
  register long r0 __asm__("r0") = arg1;
#define REGISTER_DECL_2                                                        \
  REGISTER_DECL_1                                                              \
  register long r1 __asm__("r1") = arg2;
#define REGISTER_DECL_3                                                        \
  REGISTER_DECL_2                                                              \
  register long r2 __asm__("r2") = arg3;
#define REGISTER_DECL_4                                                        \
  REGISTER_DECL_3                                                              \
  register long r3 __asm__("r3") = arg4;
#define REGISTER_DECL_5                                                        \
  REGISTER_DECL_4                                                              \
  register long r4 __asm__("r4") = arg5;
#define REGISTER_DECL_6                                                        \
  REGISTER_DECL_5                                                              \
  register long r5 __asm__("r5") = arg6;

#define REGISTER_CONSTRAINT_0 "r"(r7)
#define REGISTER_CONSTRAINT_1 REGISTER_CONSTRAINT_0, "r"(r0)
#define REGISTER_CONSTRAINT_2 REGISTER_CONSTRAINT_1, "r"(r1)
#define REGISTER_CONSTRAINT_3 REGISTER_CONSTRAINT_2, "r"(r2)
#define REGISTER_CONSTRAINT_4 REGISTER_CONSTRAINT_3, "r"(r3)
#define REGISTER_CONSTRAINT_5 REGISTER_CONSTRAINT_4, "r"(r4)
#define REGISTER_CONSTRAINT_6 REGISTER_CONSTRAINT_5, "r"(r5)

#define SYSCALL_INSTR(input_constraint)                                        \
  LIBC_INLINE_ASM("svc 0" : "=r"(r0) : input_constraint : "memory", "cc")

namespace __llvm_libc {

__attribute__((always_inline)) inline long syscall(long number) {
  REGISTER_DECL_0;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_0);
  return r0;
}

__attribute__((always_inline)) inline long syscall(long number, long arg1) {
  REGISTER_DECL_1;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_1);
  return r0;
}

__attribute__((always_inline)) inline long syscall(long number, long arg1,
                                                   long arg2) {
  REGISTER_DECL_2;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_2);
  return r0;
}

__attribute__((always_inline)) inline long syscall(long number, long arg1,
                                                   long arg2, long arg3) {
  REGISTER_DECL_3;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_3);
  return r0;
}

__attribute__((always_inline)) inline long
syscall(long number, long arg1, long arg2, long arg3, long arg4) {
  REGISTER_DECL_4;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_4);
  return r0;
}

__attribute__((always_inline)) inline long
syscall(long number, long arg1, long arg2, long arg3, long arg4, long arg5) {
  REGISTER_DECL_5;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_5);
  return r0;
}

__attribute__((always_inline)) inline long syscall(long number, long arg1,
                                                   long arg2, long arg3,
                                                   long arg4, long arg5,
                                                   long arg6) {
  REGISTER_DECL_6;
  SYSCALL_INSTR(REGISTER_CONSTRAINT_6);
  return r0;
}

} // namespace __llvm_libc

#undef REGISTER_DECL_0
#undef REGISTER_DECL_1
#undef REGISTER_DECL_2
#undef REGISTER_DECL_3
#undef REGISTER_DECL_4
#undef REGISTER_DECL_5
#undef REGISTER_DECL_6

#undef REGISTER_CONSTRAINT_0
#undef REGISTER_CONSTRAINT_1
#undef REGISTER_CONSTRAINT_2
#undef REGISTER_CONSTRAINT_3
#undef REGISTER_CONSTRAINT_4
#undef REGISTER_CONSTRAINT_5
#undef REGISTER_CONSTRAINT_6

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_ARM_SYSCALL_H
