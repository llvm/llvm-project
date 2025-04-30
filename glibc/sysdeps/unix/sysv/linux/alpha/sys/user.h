/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _SYS_USER_H
#define _SYS_USER_H	1

/* The whole purpose of this file is for gdb/strace and gdb/strace
   only. Don't read too much into it. Don't use it for anything other
   than gdb/strace unless you know what you are doing. */

#include <asm/reg.h>
#include <stddef.h>

struct user
{
  unsigned long	int regs[EF_SIZE / 8 + 32];	/* integer and fp regs */
  size_t u_tsize;				/* text size (pages) */
  size_t u_dsize;				/* data size (pages) */
  size_t u_ssize;				/* stack size (pages) */
  unsigned long	int start_code;			/* text starting address */
  unsigned long	int start_data;			/* data starting address */
  unsigned long	int start_stack;		/* stack starting address */
  long int signal;				/* signal causing core dump */
  struct regs *u_ar0;				/* help gdb find registers */
  unsigned long	int magic;			/* identifies a core file */
  char u_comm[32];				/* user command name */
};

#define PAGE_SHIFT		13
#define PAGE_SIZE		(1UL << PAGE_SHIFT)
#define PAGE_MASK		(~(PAGE_SIZE-1))
#define NBPG			PAGE_SIZE
#define UPAGES			1
#define HOST_TEXT_START_ADDR	(u.start_code)
#define HOST_DATA_START_ADDR	(u.start_data)
#define HOST_STACK_END_ADDR	(u.start_stack + u.u_ssize * NBPG)

#endif	/* sys/user.h */
