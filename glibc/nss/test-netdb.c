/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1998.

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

/*
  Testing of some network related lookup functions.
  The system databases looked up are:
  - /etc/services
  - /etc/hosts
  - /etc/networks
  - /etc/protocols
  The tests try to be fairly generic and simple so that they work on
  every possible setup (and might therefore not detect some possible
  errors).
*/

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include "nss.h"

#include <support/support.h>

/*
  The following define is necessary for glibc 2.0.6
*/
#ifndef INET6_ADDRSTRLEN
# define INET6_ADDRSTRLEN 46
#endif

int error_count;

static void
output_servent (const char *call, struct servent *sptr)
{
  char **pptr;

  if (sptr == NULL)
    printf ("Call: %s returned NULL\n", call);
  else
    {
      printf ("Call: %s, returned: s_name: %s, s_port: %d, s_proto: %s\n",
	      call, sptr->s_name, ntohs(sptr->s_port), sptr->s_proto);
      for (pptr = sptr->s_aliases; *pptr != NULL; pptr++)
	printf ("  alias: %s\n", *pptr);
    }
}


static void
test_services (void)
{
  struct servent *sptr;

  sptr = getservbyname ("domain", "tcp");
  output_servent ("getservbyname (\"domain\", \"tcp\")", sptr);

  sptr = getservbyname ("domain", "udp");
  output_servent ("getservbyname (\"domain\", \"udp\")", sptr);

  sptr = getservbyname ("domain", NULL);
  output_servent ("getservbyname (\"domain\", NULL)", sptr);

  sptr = getservbyname ("not-existant", NULL);
  output_servent ("getservbyname (\"not-existant\", NULL)", sptr);

  /* This shouldn't return anything.  */
  sptr = getservbyname ("", "");
  output_servent ("getservbyname (\"\", \"\")", sptr);

  sptr = getservbyname ("", "tcp");
  output_servent ("getservbyname (\"\", \"tcp\")", sptr);

  sptr = getservbyport (htons(53), "tcp");
  output_servent ("getservbyport (htons(53), \"tcp\")", sptr);

  sptr = getservbyport (htons(53), NULL);
  output_servent ("getservbyport (htons(53), NULL)", sptr);

  sptr = getservbyport (htons(1), "udp"); /* shouldn't exist */
  output_servent ("getservbyport (htons(1), \"udp\")", sptr);

  setservent (0);
  do
    {
      sptr = getservent ();
      output_servent ("getservent ()", sptr);
    }
  while (sptr != NULL);
  endservent ();
}


static void
output_hostent (const char *call, struct hostent *hptr)
{
  char **pptr;
  char buf[INET6_ADDRSTRLEN];

  if (hptr == NULL)
    printf ("Call: %s returned NULL\n", call);
  else
    {
      printf ("Call: %s returned: name: %s, addr_type: %d\n",
	      call, hptr->h_name, hptr->h_addrtype);
      if (hptr->h_aliases)
	for (pptr = hptr->h_aliases; *pptr != NULL; pptr++)
	  printf ("  alias: %s\n", *pptr);

      for (pptr = hptr->h_addr_list; *pptr != NULL; pptr++)
	printf ("  ip: %s\n",
		inet_ntop (hptr->h_addrtype, *pptr, buf, sizeof (buf)));
    }
}

static void
test_hosts (void)
{
  struct hostent *hptr1, *hptr2;
  char *name = NULL;
  size_t namelen = 0;
  struct in_addr ip;

  hptr1 = gethostbyname ("localhost");
  hptr2 = gethostbyname ("LocalHost");
  if (hptr1 != NULL || hptr2 != NULL)
    {
      if (hptr1 == NULL)
	{
	  printf ("localhost not found - but LocalHost found:-(\n");
	  ++error_count;
	}
      else if (hptr2 == NULL)
	{
	  printf ("LocalHost not found - but localhost found:-(\n");
	  ++error_count;
	}
      else if (strcmp (hptr1->h_name, hptr2->h_name) != 0)
	{
	  printf ("localhost and LocalHost have different canoncial name\n");
	  printf ("gethostbyname (\"localhost\")->%s\n", hptr1->h_name);
	  printf ("gethostbyname (\"LocalHost\")->%s\n", hptr2->h_name);
	  ++error_count;
	}
      else
	output_hostent ("gethostbyname(\"localhost\")", hptr1);
    }

  hptr1 = gethostbyname ("127.0.0.1");
  output_hostent ("gethostbyname (\"127.0.0.1\")", hptr1);

  hptr1 = gethostbyname ("10.1234");
  output_hostent ("gethostbyname (\"10.1234\")", hptr1);

  hptr1 = gethostbyname2 ("localhost", AF_INET);
  output_hostent ("gethostbyname2 (\"localhost\", AF_INET)", hptr1);

  while (gethostname (name, namelen) < 0 && errno == ENAMETOOLONG)
    {
      namelen += 2;		/* tiny increments to test a lot */
      name = xrealloc (name, namelen);
    }
  if (gethostname (name, namelen) == 0)
    {
      printf ("Hostname: %s\n", name);
      if (name != NULL)
	{
	  hptr1 = gethostbyname (name);
	  output_hostent ("gethostbyname (gethostname(...))", hptr1);
	}
    }

  ip.s_addr = htonl (INADDR_LOOPBACK);
  hptr1 = gethostbyaddr ((char *) &ip, sizeof (ip), AF_INET);
  if (hptr1 != NULL)
    {
      printf ("official name of 127.0.0.1: %s\n", hptr1->h_name);
    }

  sethostent (0);
  do
    {
      hptr1 = gethostent ();
      output_hostent ("gethostent ()", hptr1);
    }
  while (hptr1 != NULL);
  endhostent ();

}


static void
output_netent (const char *call, struct netent *nptr)
{
  char **pptr;

  if (nptr == NULL)
    printf ("Call: %s returned NULL\n", call);
  else
    {
      struct in_addr ip;

      ip.s_addr = htonl(nptr->n_net);
      printf ("Call: %s, returned: n_name: %s, network_number: %s\n",
	      call, nptr->n_name, inet_ntoa (ip));

      for (pptr = nptr->n_aliases; *pptr != NULL; pptr++)
	printf ("  alias: %s\n", *pptr);
    }
}

static void
test_network (void)
{
  struct netent *nptr;
  uint32_t ip;

  /*
     This test needs the following line in /etc/networks:
     loopback        127.0.0.0
  */
  nptr = getnetbyname ("loopback");
  output_netent ("getnetbyname (\"loopback\")",nptr);

  nptr = getnetbyname ("LoopBACK");
  output_netent ("getnetbyname (\"LoopBACK\")",nptr);

  ip = inet_network ("127.0.0.0");
  nptr = getnetbyaddr (ip, AF_INET);
  output_netent ("getnetbyaddr (inet_network (\"127.0.0.0\"), AF_INET)",nptr);

  setnetent (0);
  do
    {
      nptr = getnetent ();
      output_netent ("getnetent ()", nptr);
    }
  while (nptr != NULL);
  endnetent ();
}


static void
output_protoent (const char *call, struct protoent *prptr)
{
  char **pptr;

  if (prptr == NULL)
    printf ("Call: %s returned NULL\n", call);
  else
    {
      printf ("Call: %s, returned: p_name: %s, p_proto: %d\n",
	      call, prptr->p_name, prptr->p_proto);
      for (pptr = prptr->p_aliases; *pptr != NULL; pptr++)
	printf ("  alias: %s\n", *pptr);
    }
}


static void
test_protocols (void)
{
  struct protoent *prptr;

  prptr = getprotobyname ("IP");
  output_protoent ("getprotobyname (\"IP\")", prptr);

  prptr = getprotobynumber (1);
  output_protoent ("getprotobynumber (1)", prptr);

  setprotoent (0);
  do
    {
      prptr = getprotoent ();
      output_protoent ("getprotoent ()", prptr);
    }
  while (prptr != NULL);
  endprotoent ();
}


/* Override /etc/nsswitch.conf for this program.  This is mainly
   useful for developers. */
static void  __attribute__ ((unused))
setdb (const char *dbname)
{
  if (strcmp ("db", dbname))
      {
	/*
	  db is not implemented for hosts, networks
	*/
	__nss_configure_lookup ("hosts", dbname);
	__nss_configure_lookup ("networks", dbname);
      }
  __nss_configure_lookup ("protocols", dbname);
  __nss_configure_lookup ("services", dbname);
}


static int
do_test (void)
{
  /*
    setdb ("db");
  */

  test_hosts ();
  test_network ();
  test_protocols ();
  test_services ();

  if (error_count)
    printf ("\n %d errors occurred!\n", error_count);
  else
    printf ("No visible errors occurred!\n");

  return (error_count != 0);
}

#include <support/test-driver.c>
