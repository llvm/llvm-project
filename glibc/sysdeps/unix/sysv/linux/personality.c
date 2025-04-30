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

#include <sys/personality.h>
#include <sysdep.h>

extern __typeof (personality) __personality;

int
__personality (unsigned long persona)
{
#ifdef PERSONALITY_TRUNCATE_ARGUMENT
  /* Starting with kernel commit v2.6.21-3117-g97dc32c, the type of
     task_struct->pesonality is "unsigned int".
     Starting with kernel commit v2.6.35-rc1-372-g485d527, the personality
     syscall accepts "unsigned int" instead of "long unsigned int".
     Inbetween, a personality argument that does not fit into "unsigned int"
     would result to system call returning -EINVAL.
     We explicitly truncate the personality argument to "unsigned int"
     to eliminate the uncertainty.  */
  persona = (unsigned int) persona;
#endif

  long int ret = INTERNAL_SYSCALL_CALL (personality, persona);

  /* Starting with kernel commit v2.6.29-6609-g11d06b2, the personality syscall
     never fails.  However, 32-bit kernels might flag valid values as errors, so
     we need to reverse the error setting.  We can't use the raw result as some
     arches split the return/error values.  */
  if (__glibc_unlikely (INTERNAL_SYSCALL_ERROR_P (ret)))
    ret = -INTERNAL_SYSCALL_ERRNO (ret);
  return ret;
}
weak_alias (__personality, personality)
