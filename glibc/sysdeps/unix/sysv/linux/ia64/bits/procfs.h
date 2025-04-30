/* Types for registers for sys/procfs.h.  IA64 version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _SYS_PROCFS_H
# error "Never include <bits/procfs.h> directly; use <sys/procfs.h> instead."
#endif

#include <signal.h>
#include <sys/ucontext.h>
#include <bits/sigcontext.h>

/* We really need just 72 but let's leave some headroom...  */
#define ELF_NGREG	128
/* f0 and f1 could be omitted, but so what...  */
#define ELF_NFPREG	128

typedef unsigned long elf_greg_t;
typedef elf_greg_t elf_gregset_t[ELF_NGREG];

typedef struct __ia64_fpreg elf_fpreg_t;
typedef elf_fpreg_t elf_fpregset_t[ELF_NFPREG];

typedef elf_greg_t greg_t;
typedef elf_gregset_t gregset_t;
typedef elf_fpregset_t fpregset_t;
#define NGREG ELF_NGREG
