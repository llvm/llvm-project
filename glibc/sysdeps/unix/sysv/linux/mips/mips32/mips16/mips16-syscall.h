/* MIPS16 syscall wrappers.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#ifndef MIPS16_SYSCALL_H
#define MIPS16_SYSCALL_H 1

long long int __nomips16 __mips16_syscall0 (long int number);
#define __mips16_syscall0(dummy, number)				\
	__mips16_syscall0 ((long int) (number))

long long int __nomips16 __mips16_syscall1 (long int a0,
					    long int number);
#define __mips16_syscall1(a0, number)					\
	__mips16_syscall1 ((long int) (a0),				\
			   (long int) (number))

long long int __nomips16 __mips16_syscall2 (long int a0, long int a1,
					    long int number);
#define __mips16_syscall2(a0, a1, number)				\
	__mips16_syscall2 ((long int) (a0), (long int) (a1),		\
			   (long int) (number))

long long int __nomips16 __mips16_syscall3 (long int a0, long int a1,
					    long int a2,
					    long int number);
#define __mips16_syscall3(a0, a1, a2, number)				\
	__mips16_syscall3 ((long int) (a0), (long int) (a1),		\
			   (long int) (a2),				\
			   (long int) (number))

long long int __nomips16 __mips16_syscall4 (long int a0, long int a1,
					    long int a2, long int a3,
					    long int number);
#define __mips16_syscall4(a0, a1, a2, a3, number)			\
	__mips16_syscall4 ((long int) (a0), (long int) (a1),		\
			   (long int) (a2), (long int) (a3),		\
			   (long int) (number))

/* The remaining ones use regular MIPS wrappers.  */

#define __mips16_syscall5(a0, a1, a2, a3, a4, number)			\
	__mips_syscall5 ((long int) (a0), (long int) (a1),		\
			 (long int) (a2), (long int) (a3),		\
			 (long int) (a4),				\
			 (long int) (number))

#define __mips16_syscall6(a0, a1, a2, a3, a4, a5, number)		\
	__mips_syscall6 ((long int) (a0), (long int) (a1),		\
			 (long int) (a2), (long int) (a3),		\
			 (long int) (a4), (long int) (a5),		\
			 (long int) (number))

#define __mips16_syscall7(a0, a1, a2, a3, a4, a5, a6, number)		\
	__mips_syscall7 ((long int) (a0), (long int) (a1),		\
			 (long int) (a2), (long int) (a3),		\
			 (long int) (a4), (long int) (a5),		\
			 (long int) (a6),				\
			 (long int) (number))

#endif
