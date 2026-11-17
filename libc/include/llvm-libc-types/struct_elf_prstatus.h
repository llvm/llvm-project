//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct elf_prstatus.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_ELF_PRSTATUS_H
#define LLVM_LIBC_TYPES_STRUCT_ELF_PRSTATUS_H

#include "elf_gregset_t.h"
#include "pid_t.h"
#include "struct_elf_siginfo.h"
#include "struct_timeval.h"

// The NT_PRSTATUS core dump note. The layout matches the kernel's
// struct elf_prstatus (struct elf_prstatus_common followed by pr_reg and
// pr_fpvalid) in include/linux/elfcore.h, with the common fields flattened.
struct elf_prstatus {
  struct elf_siginfo pr_info;
  short pr_cursig;
  unsigned long pr_sigpend;
  unsigned long pr_sighold;
  pid_t pr_pid;
  pid_t pr_ppid;
  pid_t pr_pgrp;
  pid_t pr_sid;
  struct timeval pr_utime;
  struct timeval pr_stime;
  struct timeval pr_cutime;
  struct timeval pr_cstime;
  elf_gregset_t pr_reg;
  int pr_fpvalid;
};

#endif // LLVM_LIBC_TYPES_STRUCT_ELF_PRSTATUS_H
