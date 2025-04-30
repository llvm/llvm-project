/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_USER_H
#define _SYS_USER_H	1

#include <asm/ptrace.h>
#include <stddef.h>

/* asm/ptrace.h polutes the namespace.  */
#undef PTRACE_GETREGS
#undef PTRACE_SETREGS
#undef PTRACE_GETFPREGS
#undef PTRACE_SETFPREGS
#undef PTRACE_GETFDPIC
#undef PTRACE_GETFDPIC_EXEC
#undef PTRACE_GETFDPIC_INTERP
#undef	PTRACE_GETDSPREGS
#undef	PTRACE_SETDSPREGS

typedef unsigned long elf_greg_t;

#define ELF_NGREG (sizeof (struct pt_regs) / sizeof (elf_greg_t))
typedef elf_greg_t elf_gregset_t[ELF_NGREG];

struct user_fpu_struct
  {
    unsigned long fp_regs[16];
    unsigned long xfp_regs[16];
    unsigned long fpscr;
    unsigned long fpul;
  };
typedef struct user_fpu_struct elf_fpregset_t;

struct user
  {
    struct pt_regs regs;
    struct user_fpu_struct fpu;
    int u_fpvalid;
    size_t u_tsize;
    size_t u_dsize;
    size_t u_ssize;
    unsigned long start_code;
    unsigned long start_data;
    unsigned long start_stack;
    long int signal;
    unsigned long u_ar0;
    struct user_fpu_struct *u_fpstate;
    unsigned long magic;
    char u_comm[32];
  };

#endif  /* sys/user.h */
