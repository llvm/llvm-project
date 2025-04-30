/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>.

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

#include <errno.h>
#include <sysdep.h>
#include <setjmp.h>
#include <bits/setjmp.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <signal.h>
#include <sys/syscall.h>

#define __longjmp ____longjmp_chk

#define CHECK_SP(env, guard) \
  do									\
    {									\
      uintptr_t cur_sp;							\
      uintptr_t new_sp = env->__gregs[9];				\
      __asm__ ("lr %0, %%r15" : "=r" (cur_sp));				\
      new_sp ^= guard;							\
      if (new_sp < cur_sp)						\
	{								\
	  stack_t oss;							\
	  int res = INTERNAL_SYSCALL_CALL (sigaltstack, NULL, &oss);	\
	  if (!INTERNAL_SYSCALL_ERROR_P (res))				\
	    {								\
	      if ((oss.ss_flags & SS_ONSTACK) == 0			\
		  || ((uintptr_t) (oss.ss_sp + oss.ss_size) - new_sp	\
		      < oss.ss_size))					\
		__fortify_fail ("longjmp causes uninitialized stack frame");\
	    }								\
	}								\
    } while (0)

#include "__longjmp.c"
