/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#define SA_RESTORER	0x04000000

extern void __default_sa_restorer (void);
extern void __default_rt_sa_restorer (void);

#define SET_SA_RESTORER(kact, act)				\
 ({								\
   if ((kact)->sa_flags & SA_RESTORER)				\
     (kact)->sa_restorer = (act)->sa_restorer;			\
   else								\
     {								\
       (kact)->sa_restorer = ((kact)->sa_flags & SA_SIGINFO)	\
			     ? __default_rt_sa_restorer		\
			     : __default_sa_restorer;		\
       (kact)->sa_flags |= SA_RESTORER;				\
     }								\
 })

#define RESET_SA_RESTORER(act, kact)				\
  (act)->sa_restorer = (kact)->sa_restorer;

#include <sysdeps/unix/sysv/linux/libc_sigaction.c>
