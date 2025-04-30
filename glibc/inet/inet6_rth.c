/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2006.

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

#include <string.h>
#include <netinet/in.h>
#include <netinet/ip6.h>


/* RFC 3542, 7.1

   This function returns the number of bytes required to hold a
   Routing header of the specified type containing the specified
   number of segments (addresses).  For an IPv6 Type 0 Routing header,
   the number of segments must be between 0 and 127, inclusive.  */
socklen_t
inet6_rth_space (int type, int segments)
{
  switch (type)
    {
    case IPV6_RTHDR_TYPE_0:
      if (segments < 0 || segments > 127)
	return 0;

      return sizeof (struct ip6_rthdr0) + segments * sizeof (struct in6_addr);
    }

  return 0;
}


/* RFC 3542, 7.2

   This function initializes the buffer pointed to by BP to contain a
   Routing header of the specified type and sets ip6r_len based on the
   segments parameter.  */
void *
inet6_rth_init (void *bp, socklen_t bp_len, int type, int segments)
{
  struct ip6_rthdr *rthdr = (struct ip6_rthdr *) bp;

  switch (type)
    {
    case IPV6_RTHDR_TYPE_0:
      /* Make sure the parameters are valid and the buffer is large enough.  */
      if (segments < 0 || segments > 127)
	break;

      socklen_t len = (sizeof (struct ip6_rthdr0)
		       + segments * sizeof (struct in6_addr));
      if (len > bp_len)
	break;

      /* Some implementations seem to initialize the whole memory area.  */
      memset (bp, '\0', len);

      /* Length in units of 8 octets.  */
      rthdr->ip6r_len = segments * sizeof (struct in6_addr) / 8;
      rthdr->ip6r_type = IPV6_RTHDR_TYPE_0;
      return bp;
    }

  return NULL;
}


/* RFC 3542, 7.3

   This function adds the IPv6 address pointed to by addr to the end of
   the Routing header being constructed.  */
int
inet6_rth_add (void *bp, const struct in6_addr *addr)
{
  struct ip6_rthdr *rthdr = (struct ip6_rthdr *) bp;

  switch (rthdr->ip6r_type)
    {
      struct ip6_rthdr0 *rthdr0;
    case IPV6_RTHDR_TYPE_0:
      rthdr0 = (struct ip6_rthdr0 *) rthdr;
      if (rthdr0->ip6r0_len * 8 / sizeof (struct in6_addr)
	  - rthdr0->ip6r0_segleft < 1)
        return -1;

      memcpy (&rthdr0->ip6r0_addr[rthdr0->ip6r0_segleft++],
	      addr, sizeof (struct in6_addr));

      return 0;
    }

  return -1;
}


/* RFC 3542, 7.4

   This function takes a Routing header extension header (pointed to by
   the first argument) and writes a new Routing header that sends
   datagrams along the reverse of that route.  The function reverses the
   order of the addresses and sets the segleft member in the new Routing
   header to the number of segments.  */
int
inet6_rth_reverse (const void *in, void *out)
{
  struct ip6_rthdr *in_rthdr = (struct ip6_rthdr *) in;

  switch (in_rthdr->ip6r_type)
    {
      struct ip6_rthdr0 *in_rthdr0;
      struct ip6_rthdr0 *out_rthdr0;
    case IPV6_RTHDR_TYPE_0:
      in_rthdr0 = (struct ip6_rthdr0 *) in;
      out_rthdr0 = (struct ip6_rthdr0 *) out;

      /* Copy header, not the addresses.  The memory regions can overlap.  */
      memmove (out_rthdr0, in_rthdr0, sizeof (struct ip6_rthdr0));

      int total = in_rthdr0->ip6r0_len * 8 / sizeof (struct in6_addr);
      for (int i = 0; i < total / 2; ++i)
	{
	  /* Remember, IN_RTHDR0 and OUT_RTHDR0 might overlap.  */
	  struct in6_addr temp = in_rthdr0->ip6r0_addr[i];
	  out_rthdr0->ip6r0_addr[i] = in_rthdr0->ip6r0_addr[total - 1 - i];
	  out_rthdr0->ip6r0_addr[total - 1 - i] = temp;
	}
      if (total % 2 != 0 && in != out)
	out_rthdr0->ip6r0_addr[total / 2] = in_rthdr0->ip6r0_addr[total / 2];

      out_rthdr0->ip6r0_segleft = total;

      return 0;
    }

  return -1;
}


/* RFC 3542, 7.5

   This function returns the number of segments (addresses) contained in
   the Routing header described by BP.  */
int
inet6_rth_segments (const void *bp)
{
  struct ip6_rthdr *rthdr = (struct ip6_rthdr *) bp;

  switch (rthdr->ip6r_type)
    {
    case IPV6_RTHDR_TYPE_0:

      return rthdr->ip6r_len * 8 / sizeof (struct in6_addr);
    }

  return -1;
}


/* RFC 3542, 7.6

   This function returns a pointer to the IPv6 address specified by
   index (which must have a value between 0 and one less than the
   value returned by 'inet6_rth_segments') in the Routing header
   described by BP.  */
struct in6_addr *
inet6_rth_getaddr (const void *bp, int index)
{
  struct ip6_rthdr *rthdr = (struct ip6_rthdr *) bp;

  switch (rthdr->ip6r_type)
    {
       struct ip6_rthdr0 *rthdr0;
    case IPV6_RTHDR_TYPE_0:
      rthdr0 = (struct ip6_rthdr0 *) rthdr;

      if (index >= rthdr0->ip6r0_len * 8 / sizeof (struct in6_addr))
	break;

      return &rthdr0->ip6r0_addr[index];
    }

  return NULL;
}
