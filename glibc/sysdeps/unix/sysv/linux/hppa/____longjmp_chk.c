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

   You should have received a copy of the GNU General Public License
   along with GCC; see the file COPYING.  If not, write to the Free
   Software Foundation, 51 Franklin Street, Fifth Floor, Boston, MA
   02110-1301, USA.  */

#include <stdio.h>
#include <signal.h>
#include <sysdep.h>
#define __longjmp ____longjmp_chk
#define CHECK_SP(sp)							\
  do {									\
    register unsigned long this_sp asm ("r30");				\
    /* The stack grows up, therefore frames that were created and then	\
       destroyed must all have stack values higher than ours.  */	\
    if ((unsigned long) (sp) > this_sp)					\
      {									\
        stack_t oss;							\
        int result = INTERNAL_SYSCALL_CALL (sigaltstack, NULL, &oss);\
	/* If we aren't using an alternate stack then we have already	\
	   shown that we are jumping to a frame that doesn't exist so	\
	   error out. If we are using an alternate stack we must prove	\
	   that we are jumping *out* of the alternate stack. Note that	\
	   the check for that is the same as that for _STACK_GROWS_UP	\
	   as for _STACK_GROWS_DOWN.  */				\
        if (!INTERNAL_SYSCALL_ERROR_P (result)				\
            && ((oss.ss_flags & SS_ONSTACK) == 0			\
                || ((unsigned long) oss.ss_sp + oss.ss_size		\
                    - (unsigned long) (sp)) < oss.ss_size))		\
          __fortify_fail ("longjmp causes uninitialized stack frame");	\
      }									\
  } while (0)

#include <__longjmp.c>
