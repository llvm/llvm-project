//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of macros from sys/procfs.h.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_SYS_PROCFS_MACROS_H
#define LLVM_LIBC_MACROS_SYS_PROCFS_MACROS_H

// Number of general purpose registers in elf_gregset_t, computed the same way
// as the kernel's ELF_NGREG.
#define ELF_NGREG (sizeof(struct user_regs_struct) / sizeof(elf_greg_t))

// Size of elf_prpsinfo's pr_psargs, the kernel's ELF_PRARGSZ.
#define ELF_PRARGSZ 80

#endif // LLVM_LIBC_MACROS_SYS_PROCFS_MACROS_H
