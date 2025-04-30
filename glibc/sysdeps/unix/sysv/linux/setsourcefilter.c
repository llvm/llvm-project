/* Set source filter.  Linux version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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

#include <alloca.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include "getsourcefilter.h"


int
setsourcefilter (int s, uint32_t interface, const struct sockaddr *group,
		 socklen_t grouplen, uint32_t fmode, uint32_t numsrc,
		 const struct sockaddr_storage *slist)
{
  /* We have to create an struct ip_msfilter object which we can pass
     to the kernel.  */
  size_t needed = GROUP_FILTER_SIZE (numsrc);
  int use_alloca = __libc_use_alloca (needed);

  struct group_filter *gf;
  if (use_alloca)
    gf = (struct group_filter *) alloca (needed);
  else
    {
      gf = (struct group_filter *) malloc (needed);
      if (gf == NULL)
	return -1;
    }

  gf->gf_interface = interface;
  memcpy (&gf->gf_group, group, grouplen);
  gf->gf_fmode = fmode;
  gf->gf_numsrc = numsrc;
  memcpy (gf->gf_slist, slist, numsrc * sizeof (struct sockaddr_storage));

  /* We need to provide the appropriate socket level value.  */
  int result;
  int sol = __get_sol (group->sa_family, grouplen);
  if (sol == -1)
    {
      __set_errno (EINVAL);
      result = -1;
    }
  else
    result = __setsockopt (s, sol, MCAST_MSFILTER, gf, needed);

  if (! use_alloca)
    {
      int save_errno = errno;
      free (gf);
      __set_errno (save_errno);
    }

  return result;
}
