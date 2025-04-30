/* Copyright (C) 2009-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _SYS_USER_H
# define _SYS_USER_H	1

/* The whole purpose of this file is for GDB and GDB only.  Don't read
   too much into it.  Don't use it for anything other than GDB unless
   you know what you are doing.  */

struct user_fpregs_struct
{
  long int cwd;
  long int swd;
  long int twd;
  long int fip;
  long int fcs;
  long int foo;
  long int fos;
  long int st_space [20];
};

struct user_regs_struct
{
  unsigned int gpr[32];
  unsigned int pc;
  unsigned int msr;
  unsigned int ear;
  unsigned int esr;
  unsigned int fsr;
  unsigned int btr;
  unsigned int pvr[12];
};

struct user
{
  struct user_regs_struct	regs;
  int				u_fpvalid;
  struct user_fpregs_struct	elf_fpregset_t;
  unsigned long int		u_tsize;
  unsigned long int		u_dsize;
  unsigned long int		u_ssize;
  unsigned long			start_code;
  unsigned long			start_stack;
  long int			signal;
  int				reserved;
  struct user_regs_struct*	u_ar0;
  struct user_fpregs_struct*	u_fpstate;
  unsigned long int		magic;
  char				u_comm [32];
  int				u_debugreg [8];
};

#endif	/* _SYS_USER_H */
