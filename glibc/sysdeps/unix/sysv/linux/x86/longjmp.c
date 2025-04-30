/* __libc_siglongjmp for Linux/x86
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <sysdeps/x86/longjmp.c>

#include <pthreadP.h>
#include <jmp_buf-ssp.h>

#ifdef __x86_64__
# define SHADOW_STACK_POINTER_SIZE 8
#else
# define SHADOW_STACK_POINTER_SIZE 4
#endif

/* Assert that the priv field in struct pthread_unwind_buf has space
   to store shadow stack pointer.  */
_Static_assert ((offsetof (struct pthread_unwind_buf, priv)
                <= SHADOW_STACK_POINTER_OFFSET)
               && ((offsetof (struct pthread_unwind_buf, priv)
                    + sizeof (((struct pthread_unwind_buf *) 0)->priv))
                   >= (SHADOW_STACK_POINTER_OFFSET
                       + SHADOW_STACK_POINTER_SIZE)),
               "Shadow stack pointer is not within private storage "
               "of pthread_unwind_buf.");
