/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#undef weak_alias
#define weak_alias(a, b)
#undef strong_alias
#define strong_alias(a, b)

#include <sysdeps/unix/sysv/linux/posix_fadvise64.c>

/* Although both posix_fadvise and posix_fadvise64 has the same semantic
   on mips64, there were were releases with both symbol versions (BZ#14044).
   So we need to continue export them.  */
#if SHLIB_COMPAT(libc, GLIBC_2_2, GLIBC_2_3_3)
_strong_alias (__posix_fadvise64_l64, __posix_fadvise64_l32);
compat_symbol (libc, __posix_fadvise64_l32, posix_fadvise64, GLIBC_2_2);
versioned_symbol (libc, __posix_fadvise64_l64, posix_fadvise64, GLIBC_2_3_3);
#else
_weak_alias (posix_fadvise, posix_fadvise64);
#endif
_strong_alias (__posix_fadvise64_l64, posix_fadvise);
