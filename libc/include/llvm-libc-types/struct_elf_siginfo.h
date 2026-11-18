//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct elf_siginfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_ELF_SIGINFO_H
#define LLVM_LIBC_TYPES_STRUCT_ELF_SIGINFO_H

// Matches struct elf_siginfo in the kernel's include/linux/elfcore.h.
struct elf_siginfo {
  int si_signo;
  int si_code;
  int si_errno;
};

#endif // LLVM_LIBC_TYPES_STRUCT_ELF_SIGINFO_H
