/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.

   Versioned copy of debug/longjmp_chk.c modified for versioning
   the reverted jmpbuf extension.  */

#include <shlib-compat.h>

#if IS_IN (libc) && defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_19, GLIBC_2_20)
/* this is a copy from debug/longjmp_chk.c because we need an unique name
   for __longjmp_chk, but it is already named via a define
   for __libc_siglongjmp in debug/longjmp_chk.c.  */
# include <setjmp.h>

// XXX Should move to include/setjmp.h
extern void ____longjmp_chk (__jmp_buf __env, int __val)
     __attribute__ ((__noreturn__));

# define __longjmp ____longjmp_chk
# define __libc_siglongjmp __v1__longjmp_chk

# include <setjmp/longjmp.c>

/* In glibc release 2.19 a new versions of __longjmp_chk was introduced,
   but was reverted before 2.20. Thus both versions are the same function.  */
strong_alias (__v1__longjmp_chk, __v2__longjmp_chk);
versioned_symbol (libc, __v1__longjmp_chk, __longjmp_chk, GLIBC_2_11);
compat_symbol (libc, __v2__longjmp_chk, __longjmp_chk, GLIBC_2_19);

#else

# include <debug/longjmp_chk.c>

#endif
