/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef SHARED

/* Mark symbols hidden in static PIE for early self relocation to work.  */
# if BUILD_PIE_DEFAULT
#  pragma GCC visibility push(hidden)
# endif
/* Define I386_USE_SYSENTER to support syscall during startup in static
   PIE.  */
# include <startup.h>
# include <ldsodefs.h>
# include <cpu-features.c>

extern struct cpu_features _dl_x86_cpu_features;

# define ARCH_INIT_CPU_FEATURES() init_cpu_features (&_dl_x86_cpu_features)

#endif /* !SHARED */
#include <csu/libc-start.c>
