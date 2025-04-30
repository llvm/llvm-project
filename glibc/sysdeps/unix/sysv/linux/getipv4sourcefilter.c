/* Get IPv4 source filter.  Linux version.
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
#include <sys/param.h>
#include <sys/socket.h>


int
getipv4sourcefilter (int s, struct in_addr interface, struct in_addr group,
		     uint32_t *fmode, uint32_t *numsrc, struct in_addr *slist)
{
  /* We have to create an struct ip_msfilter object which we can pass
     to the kernel.  */
  socklen_t needed = IP_MSFILTER_SIZE (*numsrc);
  int use_alloca = __libc_use_alloca (needed);

  struct ip_msfilter *imsf;
  if (use_alloca)
    imsf = (struct ip_msfilter *) alloca (needed);
  else
    {
      imsf = (struct ip_msfilter *) malloc (needed);
      if (imsf == NULL)
	return -1;
    }

  imsf->imsf_multiaddr = group;
  imsf->imsf_interface = interface;
  imsf->imsf_numsrc = *numsrc;

  int result = __getsockopt (s, SOL_IP, IP_MSFILTER, imsf, &needed);

  /* If successful, copy the results to the places the caller wants
     them in.  */
  if (result == 0)
    {
      *fmode = imsf->imsf_fmode;
      memcpy (slist, imsf->imsf_slist,
	      MIN (*numsrc, imsf->imsf_numsrc) * sizeof (struct in_addr));
      *numsrc = imsf->imsf_numsrc;
    }

  if (! use_alloca)
    {
      int save_errno = errno;
      free (imsf);
      __set_errno (save_errno);
    }

  return result;
}
