/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <signal.h>
#include <stdio.h>
#define __longjmp ____longjmp_chk
#define CHECK_SP(saved_sp, cur_sp, sp_type)				\
  do {									\
    sp_type sp_saved = (sp_type) (saved_sp);				\
    if (sp_saved < (cur_sp))						\
      {									\
	struct __jmp_buf_internal_tag *env_save = env_arg;		\
	int val_save = val_arg;						\
	stack_t ss;							\
	int ret = __sigaltstack (NULL, &ss);				\
	if (ret == 0							\
	    && (!(ss.ss_flags & SS_ONSTACK)				\
		|| ((unsigned sp_type) ((sp_type) (long) ss.ss_sp	\
					+ (sp_type) ss.ss_size		\
					- sp_saved)			\
		    < ss.ss_size)))					\
	  __fortify_fail ("longjmp causes uninitialized stack frame");	\
	asm volatile ("move %0, %1" : "=r" (env) : "r" (env_save));	\
	asm volatile ("move %0, %1" : "=r" (val) : "r" (val_save));	\
      }									\
  } while (0)
#include <__longjmp.c>
