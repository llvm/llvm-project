/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger <davidm@hpl.hp.com>

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

/* __pipe is a special syscall since it returns two values.  */

#include <sysdep.h>

ENTRY(__pipe)
       .regstk 1,0,0,0
       DO_CALL (SYS_ify (pipe))
       cmp.ne p6,p0=-1,r10
       ;;
(p6)   st4 [in0]=r8,4
(p6)   mov ret0=0
       ;;
(p6)   st4 [in0]=r9
(p6)   ret
       br.cond.spnt.few __syscall_error
PSEUDO_END(__pipe)

libc_hidden_def (__pipe)
weak_alias (__pipe, pipe)
