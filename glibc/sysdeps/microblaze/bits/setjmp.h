/* Copyright (C) 1997-2021 Free Software Foundation, Inc.

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

/* Define the machine-dependent type `jmp_buf'.  */

#ifndef _BITS_SETJMP_H
# define _BITS_SETJMP_H 1

#if !defined _SETJMP_H && !defined _PTHREAD_H
# error "Never include <bits/setjmp.h> directly; use <setjmp.h> instead."
#endif

typedef struct __jmp_buf_internal_tag
  {
    /* There are 21 4-byte registers that should be saved:
       r1, r2, r13-r31. Actually, there seems no need to save
       r14, r16, r17, r18 (return addresses for interrupt/exception/trap).  */
    int *__sp; /* dedicated name for r1.  */
    long int __gregs[20];
  } __jmp_buf[1];

#endif
