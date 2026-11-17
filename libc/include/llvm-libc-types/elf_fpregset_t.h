//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of elf_fpregset_t type.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_ELF_FPREGSET_T_H
#define LLVM_LIBC_TYPES_ELF_FPREGSET_T_H

// The floating-point register set the kernel writes to an NT_PRFPREG core
// dump note (see elf_core_copy_task_fpregs in the kernel).
#if defined(__x86_64__)
#include "struct_user_fpregs_struct.h"
typedef struct user_fpregs_struct elf_fpregset_t;
#elif defined(__aarch64__)
#include <asm/ptrace.h>
typedef struct user_fpsimd_state elf_fpregset_t;
#else
#error "elf_fpregset_t not available for your target architecture."
#endif

#endif // LLVM_LIBC_TYPES_ELF_FPREGSET_T_H
