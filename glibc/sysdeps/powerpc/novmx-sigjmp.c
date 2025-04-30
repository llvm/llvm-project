/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

/* Copy of sysdeps/generic/sigjmp.c modified for backward compatibility
   with old non AltiVec/VMX setjmp.  */

#include <bits/wordsize.h>
#include <shlib-compat.h>
#if IS_IN (libc) && defined SHARED
# if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4)
#  include <stddef.h>
#  include <novmxsetjmp.h>
#  include <signal.h>

/* This function is called by the `sigsetjmp' macro
   before doing a `__setjmp' on ENV[0].__jmpbuf.
   Always return zero.  */

int
__novmx__sigjmp_save (__novmx__sigjmp_buf env, int savemask)
{
  env[0].__mask_was_saved = (savemask
			     && __sigprocmask (SIG_BLOCK, (sigset_t *) NULL,
					       &env[0].__saved_mask) == 0);

  return 0;
}

# endif /* SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4) */
#endif /* IS_IN (libc) && SHARED  */
