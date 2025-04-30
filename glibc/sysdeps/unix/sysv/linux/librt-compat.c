/* Placeholder definitions to pull in removed symbol versions.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <rt/librt-compat.c>
#include <kernel-posix-timers.h>

/* GLIBC_2.3.3 symbols were added for the int -> timer_t ABI transition.  */
#if TIMER_T_WAS_INT_COMPAT
compat_symbol (librt, __librt_version_placeholder_1,
               __librt_version_placeholder, GLIBC_2_3_3);
#endif
