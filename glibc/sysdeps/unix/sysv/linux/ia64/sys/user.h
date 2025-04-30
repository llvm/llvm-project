/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <features.h>
#include <sys/types.h>

/* This definition comes directly from the kernel headers.  If
   anything changes in them this header has to be changed, too.  */


/* The definition in the kernel has the comment "XXX fix me".  */
#define EF_SIZE		3072


struct user
{
  unsigned long int regs[EF_SIZE / 8 + 32];	/* Integer and fp regs.  */
  size_t u_tsize;				/* Text size (pages).  */
  size_t u_dsize;				/* Data size (pages).  */
  size_t u_ssize;				/* Stack size (pages).  */
  unsigned long int start_code;			/* Text starting address.  */
  unsigned long int start_data;			/* Data starting address.  */
  unsigned long int start_stack;		/* Stack starting address.  */
  long int signal;				/* Signal causing core dump. */
  struct regs *u_ar0;				/* Help gdb find registers.  */
  unsigned long int magic;			/* Identifies a core file.  */
  char u_comm[32];				/* User command name.  */
};

#endif	/* sys/user.h */
