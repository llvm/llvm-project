/* Test that nsswitch.conf reloading actually works.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>
#include <pwd.h>

#include <support/support.h>
#include <support/check.h>

#include "nss_test.h"

/* Size of buffers used by *_r functions.  */
#define TESTBUFLEN 4096

static struct passwd pwd_table_1[] = {
    PWD (100),
    PWD (30),
    PWD (200),
    PWD (60),
    PWD (20000),
    PWD_LAST ()
  };

static const char *hostaddr_5[] =
  {
   "ABCD", "abcd", "1234", NULL
  };

static const char *hostaddr_15[] =
  {
   "4321", "ghij", NULL
  };

static const char *hostaddr_25[] =
  {
   "WXYZ", NULL
  };


static struct hostent host_table_1[] = {
  HOST (5),
  HOST (15),
  HOST (25),
  HOST_LAST ()
};

void
_nss_test1_init_hook(test_tables *t)
{
  t->pwd_table = pwd_table_1;
  t->host_table = host_table_1;
}

/* The first of these must not appear in pwd_table_1.  */
static struct passwd pwd_table_2[] = {
    PWD (5),
    PWD_N(200, "name30"),
    PWD (16),
    PWD_LAST ()
  };

static const char *hostaddr_6[] =
  {
   "mnop", NULL
  };

static const char *hostaddr_16[] =
  {
   "7890", "a1b2", NULL
  };

static const char *hostaddr_26[] =
  {
   "qwer", "tyui", NULL
  };

static struct hostent host_table_2[] = {
  HOST (6),
  HOST (16),
  HOST (26),
  HOST_LAST ()
};

void
_nss_test2_init_hook(test_tables *t)
{
  t->pwd_table = pwd_table_2;
  t->host_table = host_table_2;
}

static void
must_be_tests (struct passwd *pt, struct hostent *ht)
{
  int i;
  struct hostent *h;

  struct passwd *p;
  for (i = 0; !PWD_ISLAST (&pt[i]); ++i)
    {
      p = getpwuid (pt[i].pw_uid);
      TEST_VERIFY (p != NULL);
      if (p != NULL)
	{
	  TEST_VERIFY (strcmp (p->pw_name, pt[i].pw_name) == 0);
	}
    }

  setpwent ();
  for (i = 0; !PWD_ISLAST (&pt[i]); ++i)
    {
      p = getpwent ();
      TEST_VERIFY (p != NULL);
      if (p != NULL)
	{
	  TEST_VERIFY (strcmp (p->pw_name, pt[i].pw_name) == 0);
	  TEST_VERIFY (p->pw_uid == pt[i].pw_uid);
	}
    }
  endpwent ();

  for (i = 0; !HOST_ISLAST (&ht[i]); ++i)
    {
      h = gethostbyname (ht[i].h_name);
      TEST_VERIFY (h != NULL);
      if (h != NULL)
	{
	  TEST_VERIFY (strcmp (h->h_name, ht[i].h_name) == 0);
	  TEST_VERIFY (h->h_addr_list[0] != NULL);
	  if (h->h_addr_list[0])
	    TEST_VERIFY (strcmp (h->h_addr_list[0], ht[i].h_addr_list[0]) == 0);
	}
    }

  for (i = 0; !HOST_ISLAST (&ht[i]); ++i)
    {
      struct hostent r, *rp;
      char buf[TESTBUFLEN];
      int herrno, res;

      res = gethostbyname2_r (ht[i].h_name, AF_INET,
			    &r, buf, TESTBUFLEN, &rp, &herrno);
      TEST_VERIFY (res == 0);
      if (res == 0)
	{
	  TEST_VERIFY (strcmp (r.h_name, ht[i].h_name) == 0);
	  TEST_VERIFY (r.h_addr_list[0] != NULL);
	  if (r.h_addr_list[0])
	    TEST_VERIFY (strcmp (r.h_addr_list[0], ht[i].h_addr_list[0]) == 0);
	}
    }

  for (i = 0; !HOST_ISLAST (&ht[i]); ++i)
    {
      h = gethostbyaddr (ht[i].h_addr, 4, AF_INET);
      TEST_VERIFY (h != NULL);
      if (h != NULL)
	{
	  TEST_VERIFY (strcmp (h->h_name, ht[i].h_name) == 0);
	  TEST_VERIFY (h->h_addr_list[0] != NULL);
	  if (h->h_addr_list[0])
	    TEST_VERIFY (strcmp (h->h_addr_list[0], ht[i].h_addr_list[0]) == 0);
	}
    }

  /* getaddrinfo */

  for (i = 0; !HOST_ISLAST (&ht[i]); ++i)
    {
      struct addrinfo *ap;
      struct addrinfo hint;
      int res, j;

      memset (&hint, 0, sizeof (hint));
      hint.ai_family = AF_INET;
      hint.ai_socktype = SOCK_STREAM;
      hint.ai_protocol = 0;
      hint.ai_flags = 0;

      ap = NULL;
      res = getaddrinfo (ht[i].h_name, NULL, &hint, &ap);
      TEST_VERIFY (res == 0);
      TEST_VERIFY (ap != NULL);
      if (res == 0 && ap != NULL)
	{
	  j = 0; /* which address in the list */
	  while (ap)
	    {
	      struct sockaddr_in *in = (struct sockaddr_in *)ap->ai_addr;
	      unsigned char *up = (unsigned char *)&in->sin_addr;

	      TEST_VERIFY (memcmp (up, ht[i].h_addr_list[j], 4) == 0);

	      ap = ap->ai_next;
	      ++j;
	    }
	}
    }

  /* getnameinfo */

  for (i = 0; !HOST_ISLAST (&ht[i]); ++i)
    {
      struct sockaddr_in addr;
      int res;
      char host_buf[NI_MAXHOST];

      memset (&addr, 0, sizeof (addr));
      addr.sin_family = AF_INET;
      addr.sin_port = 80;
      memcpy (& addr.sin_addr, ht[i].h_addr_list[0], 4);

      res = getnameinfo ((struct sockaddr *) &addr, sizeof(addr),
			 host_buf, sizeof(host_buf),
			 NULL, 0, NI_NOFQDN);

      TEST_VERIFY (res == 0);
      if (res == 0)
	TEST_VERIFY (strcmp (ht[i].h_name, host_buf) == 0);
      else
	printf ("error %s\n", gai_strerror (res));
    }
}

static void
must_be_1 (void)
{
  struct passwd *p;

  must_be_tests (pwd_table_1, host_table_1);
  p = getpwnam("name5");
  TEST_VERIFY (p == NULL);
}

static void
must_be_2 (void)
{
  struct passwd *p;

  must_be_tests (pwd_table_2, host_table_2);
  p = getpwnam("name100");
  TEST_VERIFY (p == NULL);
}

static void
xrename (const char *a, const char *b)
{
  int i = rename (a, b);
  if (i != 0)
    FAIL_EXIT1 ("rename(%s,%s) failed: %s\n", a, b, strerror(errno));
}

/* If the actions change while in the midst of doing a series of
   lookups, make sure they're consistent.  */
static void
test_cross_switch_consistency (void)
{
  int i;
  struct passwd *p;

  /* We start by initiating a set/get/end loop on conf1.  */
  setpwent ();
  for (i = 0; !PWD_ISLAST (&pwd_table_1[i]); ++i)
    {
      p = getpwent ();
      TEST_VERIFY (p != NULL);
      if (p != NULL)
	{
	  TEST_VERIFY (strcmp (p->pw_name, pwd_table_1[i].pw_name) == 0);
	  TEST_VERIFY (p->pw_uid == pwd_table_1[i].pw_uid);
	}

      /* After the first lookup, switch to conf2 and verify */
      if (i == 0)
	{
	  xrename ("/etc/nsswitch.conf", "/etc/nsswitch.conf1");
	  xrename ("/etc/nsswitch.conf2", "/etc/nsswitch.conf");

	  p = getpwnam (pwd_table_2[0].pw_name);
	  TEST_VERIFY (p->pw_uid == pwd_table_2[0].pw_uid);
	}

      /* But the original loop should still be on conf1.  */
    }
  endpwent ();

  /* Make sure the set/get/end loop sees conf2 now.  */
  setpwent ();
  for (i = 0; !PWD_ISLAST (&pwd_table_2[i]); ++i)
    {
      p = getpwent ();
      TEST_VERIFY (p != NULL);
      if (p != NULL)
	{
	  TEST_VERIFY (strcmp (p->pw_name, pwd_table_2[i].pw_name) == 0);
	  TEST_VERIFY (p->pw_uid == pwd_table_2[i].pw_uid);
	}
    }
  endpwent ();

}

static int
do_test (void)
{
  /* The test1 module was configured at program start.  */
  must_be_1 ();

  xrename ("/etc/nsswitch.conf", "/etc/nsswitch.conf1");
  xrename ("/etc/nsswitch.conf2", "/etc/nsswitch.conf");
  must_be_2 ();

  xrename ("/etc/nsswitch.conf", "/etc/nsswitch.conf2");
  xrename ("/etc/nsswitch.conf1", "/etc/nsswitch.conf");
  must_be_1 ();

  test_cross_switch_consistency ();

  return 0;
}

#include <support/test-driver.c>
