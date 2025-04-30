/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

/* Check alignment, overlapping and layout of TLS variables.  */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <pthreadP.h>
#include <sys/param.h>

#include "tst-tls5.h"

#ifdef TLS_REGISTER

struct tls_obj tls_registry[64];

static int
tls_addr_cmp (const void *a, const void *b)
{
  if (((struct tls_obj *)a)->addr < ((struct tls_obj *)b)->addr)
    return -1;
  if (((struct tls_obj *)a)->addr > ((struct tls_obj *)b)->addr)
    return 1;
  return 0;
}

static int
do_test (void)
{
  size_t cnt, i;
  int res = 0;
  uintptr_t min_addr = ~(uintptr_t) 0, max_addr = 0;

  for (cnt = 0; tls_registry[cnt].name; ++cnt);
  tls_registry[cnt].name = NULL;
  tls_registry[cnt].addr = (uintptr_t) pthread_self ();
  tls_registry[cnt].size = sizeof (struct pthread);
  tls_registry[cnt++].align = __alignof__ (struct pthread);

  qsort (tls_registry, cnt, sizeof (struct tls_obj), tls_addr_cmp);

  for (i = 0; i < cnt; ++i)
    {
      printf ("%s%s = %p, size %zd, align %zd",
	      tls_registry[i].name ? "&" : "",
	      tls_registry[i].name ?: "pthread_self ()",
	      (void *) tls_registry[i].addr,
	      tls_registry[i].size, tls_registry[i].align);
      if (tls_registry[i].addr & (tls_registry[i].align - 1))
	{
	  fputs (", WRONG ALIGNMENT", stdout);
	  res = 1;
	}
      if (i > 0
	  && (tls_registry[i - 1].addr + tls_registry[i - 1].size
	      > tls_registry[i].addr))
	{
	  fputs (", ADDRESS OVERLAP", stdout);
	  res = 1;
	}
      puts ("");
      if (tls_registry[i].name)
	{
	  min_addr = MIN (tls_registry[i].addr, min_addr);
	  max_addr = MAX (tls_registry[i].addr + tls_registry[i].size,
			  max_addr);
	}
    }

  if (cnt > 1)
    {
#if TLS_TCB_AT_TP
      if (tls_registry[cnt - 1].name)
	{
	  puts ("pthread_self () not larger than all TLS addresses");
	  res = 1;
	}
      else
	max_addr = MAX (tls_registry[cnt - 1].addr, max_addr);
#elif TLS_DTV_AT_TP
      if (tls_registry[0].name)
	{
	  puts ("pthread_self () not smaller than all TLS addresses");
	  res = 1;
	}
#else
      abort ();
#endif
      printf ("Initial TLS used block size %zd\n",
	      (size_t) (max_addr - min_addr));
    }
  return res;
}

#define TEST_FUNCTION do_test ()

#else

#define TEST_FUNCTION 0

#endif

#include "../test-skeleton.c"
