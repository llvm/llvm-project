/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <rpc/rpc.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

static int
do_test (void)
{
  XDR xdrs;
  void *buf;
  size_t ps = sysconf (_SC_PAGESIZE);
  uintptr_t half = -1;
  int v_int;
  u_short v_u_short;

  half = (half >> 1) & ~(uintptr_t) (ps - 1);
  buf = mmap ((void *) half, 2 * ps, PROT_READ | PROT_WRITE,
	      MAP_PRIVATE | MAP_ANON, -1, 0);
  if (buf == MAP_FAILED || buf != (void *) half)
    {
      puts ("Couldn't mmap 2 pages in the middle of address space");
      return 0;
    }

  xdrmem_create (&xdrs, (char *) buf, 2 * ps, XDR_ENCODE);

#define T(type, val) \
  v_##type = val;			\
  if (! xdr_##type (&xdrs, &v_##type))	\
    {					\
      puts ("encoding of " #type	\
	    " " #val " failed");	\
      return 1;				\
    }

  T(int, 127)

  u_int pos = xdr_getpos (&xdrs);

  T(u_short, 31)

  if (! xdr_setpos (&xdrs, pos))
    {
      puts ("xdr_setpos during encoding failed");
      return 1;
    }

  T(u_short, 36)

#undef T

  xdr_destroy (&xdrs);

  xdrmem_create (&xdrs, (char *) buf, 2 * ps, XDR_DECODE);

#define T(type, val) \
  v_##type = 0x15;			\
  if (! xdr_##type (&xdrs, &v_##type))	\
    {					\
      puts ("decoding of " #type	\
	    " " #val " failed");	\
      return 1;				\
    }					\
  if (v_##type != val)			\
    {					\
      puts ("decoded value differs, "	\
	    "type " #type " " #val);	\
      return 1;				\
    }

  T(int, 127)

  pos = xdr_getpos (&xdrs);

  T(u_short, 36)

  if (! xdr_setpos (&xdrs, pos))
    {
      puts ("xdr_setpos during encoding failed");
      return 1;
    }

  T(u_short, 36)

#undef T

  xdr_destroy (&xdrs);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
