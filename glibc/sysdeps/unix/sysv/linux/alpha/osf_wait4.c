/* wait4 -- wait for process to change state.  Linux/Alpha/tv32 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <tv32-compat.h>

pid_t
attribute_compat_text_section
__wait4_tv32 (pid_t pid, int *status, int options, struct __rusage32 *usage32)
{
  struct rusage usage;
  pid_t child = __wait4 (pid, status, options, &usage);

  if (child >= 0 && usage32 != NULL)
    rusage64_to_rusage32 (&usage, usage32);
  return child;
}

compat_symbol (libc, __wait4_tv32, wait4, GLIBC_2_0);
#endif
