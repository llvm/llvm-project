/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#include <sys/resource.h>
#include <sysdep.h>

int
prlimit (__pid_t pid, enum __rlimit_resource resource,
	 const struct rlimit *new_rlimit, struct rlimit *old_rlimit)
{
  struct rlimit64 new_rlimit64_mem;
  struct rlimit64 *new_rlimit64 = NULL;
  struct rlimit64 old_rlimit64_mem;
  struct rlimit64 *old_rlimit64 = (old_rlimit != NULL
				   ? &old_rlimit64_mem : NULL);

  if (new_rlimit != NULL)
    {
      if (new_rlimit->rlim_cur == RLIM_INFINITY)
	new_rlimit64_mem.rlim_cur = RLIM64_INFINITY;
      else
	new_rlimit64_mem.rlim_cur = new_rlimit->rlim_cur;
      if (new_rlimit->rlim_max == RLIM_INFINITY)
	new_rlimit64_mem.rlim_max = RLIM64_INFINITY;
      else
	new_rlimit64_mem.rlim_max = new_rlimit->rlim_max;
      new_rlimit64 = &new_rlimit64_mem;
    }

  int res = INLINE_SYSCALL (prlimit64, 4, pid, resource, new_rlimit64,
			    old_rlimit64);

  if (res == 0 && old_rlimit != NULL)
    {
      /* The prlimit64 syscall is ill-designed for 32-bit machines.
	 We have to provide a 32-bit variant since otherwise the LFS
	 system would not work.  The infinity value can be translated,
	 but otherwise what shall we do if the syscall succeeds but the
	 old values do not fit into a rlimit structure?  We cannot return
	 an error because the operation itself worked.  Best is perhaps
	 to return RLIM_INFINITY.  */
      old_rlimit->rlim_cur = old_rlimit64_mem.rlim_cur;
      if (old_rlimit->rlim_cur != old_rlimit64_mem.rlim_cur)
	{
	  if ((new_rlimit == NULL)
	      && (old_rlimit64_mem.rlim_cur != RLIM64_INFINITY))
	    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EOVERFLOW);
	  old_rlimit->rlim_cur = RLIM_INFINITY;
	}
      old_rlimit->rlim_max = old_rlimit64_mem.rlim_max;
      if (old_rlimit->rlim_max != old_rlimit64_mem.rlim_max)
	{
	  if ((new_rlimit == NULL)
	      && (old_rlimit64_mem.rlim_max != RLIM64_INFINITY))
	    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EOVERFLOW);
	  old_rlimit->rlim_max = RLIM_INFINITY;
	}
    }

  return res;
}
