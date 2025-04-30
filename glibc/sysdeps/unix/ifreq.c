/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>.

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

#include "ifreq.h"


void
__ifreq (struct ifreq **ifreqs, int *num_ifs, int sockfd)
{
  int fd = sockfd;
  struct ifconf ifc;
  int rq_len;
  int nifs;
# define RQ_IFS	4

  if (fd < 0)
    fd = __opensock ();
  if (fd < 0)
    {
      *num_ifs = 0;
      *ifreqs = NULL;
      return;
    }

  ifc.ifc_buf = NULL;
  rq_len = RQ_IFS * sizeof (struct ifreq) / 2; /* Doubled in the loop.  */
  do
    {
      ifc.ifc_len = rq_len *= 2;
      void *newp = realloc (ifc.ifc_buf, ifc.ifc_len);
      if (newp == NULL || __ioctl (fd, SIOCGIFCONF, &ifc) < 0)
	{
	  free (ifc.ifc_buf);

	  if (fd != sockfd)
	    __close (fd);
	  *num_ifs = 0;
	  *ifreqs = NULL;
	  return;
	}
      ifc.ifc_buf = newp;
    }
  while (rq_len < sizeof (struct ifreq) + ifc.ifc_len);

  if (fd != sockfd)
    __close (fd);

#ifdef _HAVE_SA_LEN
  struct ifreq *ifr = *ifreqs;
  nifs = 0;
  while ((char *) ifr < ifc.ifc_buf + ifc.ifc_len)
    {
      ++nifs;
      ifr = __if_nextreq (ifr);
      if (ifr == NULL)
	break;
    }
#else
  nifs = ifc.ifc_len / sizeof (struct ifreq);
#endif

  *num_ifs = nifs;
  *ifreqs = realloc (ifc.ifc_buf, nifs * sizeof (struct ifreq));
}
