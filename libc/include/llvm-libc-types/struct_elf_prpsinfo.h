//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct elf_prpsinfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_ELF_PRPSINFO_H
#define LLVM_LIBC_TYPES_STRUCT_ELF_PRPSINFO_H

#include "../llvm-libc-macros/sys-procfs-macros.h"
#include "gid_t.h"
#include "pid_t.h"
#include "uid_t.h"

// The NT_PRPSINFO core dump note. The layout matches the kernel's
// struct elf_prpsinfo in include/linux/elfcore.h.
struct elf_prpsinfo {
  char pr_state;
  char pr_sname;
  char pr_zomb;
  char pr_nice;
  unsigned long pr_flag;
  uid_t pr_uid;
  gid_t pr_gid;
  pid_t pr_pid;
  pid_t pr_ppid;
  pid_t pr_pgrp;
  pid_t pr_sid;
  char pr_fname[16];
  char pr_psargs[ELF_PRARGSZ];
};

#endif // LLVM_LIBC_TYPES_STRUCT_ELF_PRPSINFO_H
