/* System-specific settings for dynamic linker code.  IA-32 version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#ifndef _LINUX_I686_DL_SYSDEP_H
#define _LINUX_I686_DL_SYSDEP_H	1

/* The i386 file does most of the work.  */
#include_next <dl-sysdep.h>

/* Actually use the vDSO entry point for syscalls.
   i386/dl-sysdep.h arranges to support it, but not use it.  */
#define USE_DL_SYSINFO	1

#endif	/* dl-sysdep.h */
