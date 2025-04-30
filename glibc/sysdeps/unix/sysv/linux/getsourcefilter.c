/* Get source filter.  Linux version.
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
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <netatalk/at.h>
#include <netax25/ax25.h>
#include <netinet/in.h>
#include <netipx/ipx.h>
#include <netpacket/packet.h>
#include <netrose/rose.h>
#include <sys/param.h>
#include <sys/socket.h>
#include "getsourcefilter.h"


static const struct
{
  int sol;
  int af;
  socklen_t size;
}  sol_map[] =
  {
    /* Sort the array according to importance of the protocols.  Add
       more protocols when they become available.  */
    { SOL_IP, AF_INET, sizeof (struct sockaddr_in) },
    { SOL_IPV6, AF_INET6, sizeof (struct sockaddr_in6) },
    { SOL_AX25, AF_AX25, sizeof (struct sockaddr_ax25) },
    { SOL_IPX, AF_IPX, sizeof (struct sockaddr_ipx) },
    { SOL_ATALK, AF_APPLETALK, sizeof (struct sockaddr_at) },
    { SOL_ROSE, AF_ROSE, sizeof (struct sockaddr_rose) },
    { SOL_PACKET, AF_PACKET, sizeof (struct sockaddr_ll) }
  };
#define NSOL_MAP (sizeof (sol_map) / sizeof (sol_map[0]))


/* Try to determine the socket level value.  Ideally both side and
   family are set.  But sometimes only the size is correct and the
   family value might be bogus.  Loop over the array entries and look
   for a perfect match or the first match based on size.  */
int
__get_sol (int af, socklen_t len)
{
  int first_size_sol = -1;

  for (size_t cnt = 0; cnt < NSOL_MAP; ++cnt)
    {
      /* Just a test so that we make sure the special value used to
	 signal the "we have so far no socket level value" is OK.  */
      assert (sol_map[cnt].sol != -1);

      if (len == sol_map[cnt].size)
	{
	  /* The size matches, which is a requirement.  If the family
	     matches, too, we have a winner.  Otherwise we remember the
	     socket level value for this protocol if it is the first
	     match.  */
	  if (af == sol_map[cnt].af)
	    /* Bingo!  */
	    return sol_map[cnt].sol;

	  if (first_size_sol == -1)
	    first_size_sol = sol_map[cnt].sol;
      }
    }

  return first_size_sol;
}


int
getsourcefilter (int s, uint32_t interface, const struct sockaddr *group,
		 socklen_t grouplen, uint32_t *fmode, uint32_t *numsrc,
		 struct sockaddr_storage *slist)
{
  /* We have to create an struct ip_msfilter object which we can pass
     to the kernel.  */
  socklen_t needed = GROUP_FILTER_SIZE (*numsrc);
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
  gf->gf_numsrc = *numsrc;

  /* We need to provide the appropriate socket level value.  */
  int result;
  int sol = __get_sol (group->sa_family, grouplen);
  if (sol == -1)
    {
      __set_errno (EINVAL);
      result = -1;
    }
  else
    {
      result = __getsockopt (s, sol, MCAST_MSFILTER, gf, &needed);

      /* If successful, copy the results to the places the caller wants
	 them in.  */
      if (result == 0)
	{
	  *fmode = gf->gf_fmode;
	  memcpy (slist, gf->gf_slist,
		  MIN (*numsrc, gf->gf_numsrc)
		  * sizeof (struct sockaddr_storage));
	  *numsrc = gf->gf_numsrc;
	}
    }

  if (! use_alloca)
    {
      int save_errno = errno;
      free (gf);
      __set_errno (save_errno);
    }

  return result;
}
