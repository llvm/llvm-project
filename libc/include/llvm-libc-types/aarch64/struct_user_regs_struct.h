//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct user_regs_struct for aarch64.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_AARCH64_STRUCT_USER_REGS_STRUCT_H
#define LLVM_LIBC_TYPES_AARCH64_STRUCT_USER_REGS_STRUCT_H

// Same layout as the kernel's struct user_pt_regs in <asm/ptrace.h>.
struct user_regs_struct {
  unsigned long long regs[31];
  unsigned long long sp;
  unsigned long long pc;
  unsigned long long pstate;
};

#endif // LLVM_LIBC_TYPES_AARCH64_STRUCT_USER_REGS_STRUCT_H
