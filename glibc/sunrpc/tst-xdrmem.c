/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2005.

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

static int
do_test (void)
{
  XDR xdrs;
  unsigned char buf[8192];
  int v_int;
  u_int v_u_int;
  long v_long;
  u_long v_u_long;
  quad_t v_hyper;
  u_quad_t v_u_hyper;
  quad_t v_longlong_t;
  u_quad_t v_u_longlong_t;
  short v_short;
  u_short v_u_short;
  char v_char;
  u_char v_u_char;
  bool_t v_bool;
  enum_t v_enum;
  char *v_wrapstring;

  xdrmem_create (&xdrs, (char *) buf, sizeof (buf), XDR_ENCODE);

#define TESTS \
  T(int, 0)				\
  T(int, CHAR_MAX)			\
  T(int, CHAR_MIN)			\
  T(int, SHRT_MAX)			\
  T(int, SHRT_MIN)			\
  T(int, INT_MAX)			\
  T(int, INT_MIN)			\
  T(int, 0x123)				\
  T(u_int, 0)				\
  T(u_int, UCHAR_MAX)			\
  T(u_int, USHRT_MAX)			\
  T(u_int, UINT_MAX)			\
  T(u_int, 0xdeadbeef)			\
  T(u_int, 0x12345678)			\
  T(long, 0)				\
  T(long, 2147483647L)			\
  T(long, -2147483648L)			\
  T(long, -305419896L)			\
  T(long, -305419896L)			\
  T(u_long, 0)				\
  T(u_long, 0xffffffffUL)		\
  T(u_long, 0xdeadbeefUL)		\
  T(u_long, 0x12345678UL)		\
  T(hyper, 0)				\
  T(hyper, CHAR_MAX)			\
  T(hyper, CHAR_MIN)			\
  T(hyper, SHRT_MAX)			\
  T(hyper, SHRT_MIN)			\
  T(hyper, INT_MAX)			\
  T(hyper, INT_MIN)			\
  T(hyper, LONG_MAX)			\
  T(hyper, LONG_MIN)			\
  T(hyper, LONG_LONG_MAX)		\
  T(hyper, LONG_LONG_MIN)		\
  T(hyper, 0x12312345678LL)		\
  T(hyper, 0x12387654321LL)		\
  T(u_hyper, 0)				\
  T(u_hyper, UCHAR_MAX)			\
  T(u_hyper, USHRT_MAX)			\
  T(u_hyper, UINT_MAX)			\
  T(u_hyper, ULONG_MAX)			\
  T(u_hyper, ULONG_LONG_MAX)		\
  T(u_hyper, 0xdeadbeefdeadbeefULL)	\
  T(u_hyper, 0x12312345678ULL)		\
  T(u_hyper, 0x12387654321ULL)		\
  T(longlong_t, 0)			\
  T(longlong_t, CHAR_MAX)		\
  T(longlong_t, CHAR_MIN)		\
  T(longlong_t, SHRT_MAX)		\
  T(longlong_t, SHRT_MIN)		\
  T(longlong_t, INT_MAX)		\
  T(longlong_t, INT_MIN)		\
  T(longlong_t, LONG_MAX)		\
  T(longlong_t, LONG_MIN)		\
  T(longlong_t, LONG_LONG_MAX)		\
  T(longlong_t, LONG_LONG_MIN)		\
  T(longlong_t, 0x12312345678LL)	\
  T(longlong_t, 0x12387654321LL)	\
  T(u_longlong_t, 0)			\
  T(u_longlong_t, UCHAR_MAX)		\
  T(u_longlong_t, USHRT_MAX)		\
  T(u_longlong_t, UINT_MAX)		\
  T(u_longlong_t, ULONG_MAX)		\
  T(u_longlong_t, ULONG_LONG_MAX)	\
  T(u_longlong_t, 0xdeadbeefdeadbeefULL)\
  T(u_longlong_t, 0x12312345678ULL)	\
  T(u_longlong_t, 0x12387654321ULL)	\
  T(short, CHAR_MAX)			\
  T(short, CHAR_MIN)			\
  T(short, SHRT_MAX)			\
  T(short, SHRT_MIN)			\
  T(short, 0x123)			\
  T(u_short, 0)				\
  T(u_short, UCHAR_MAX)			\
  T(u_short, USHRT_MAX)			\
  T(u_short, 0xbeef)			\
  T(u_short, 0x5678)			\
  T(char, CHAR_MAX)			\
  T(char, CHAR_MIN)			\
  T(char, 0x23)				\
  T(u_char, 0)				\
  T(u_char, UCHAR_MAX)			\
  T(u_char, 0xef)			\
  T(u_char, 0x78)			\
  T(bool, 0)				\
  T(bool, 1)				\
  T(enum, 0)				\
  T(enum, CHAR_MAX)			\
  T(enum, CHAR_MIN)			\
  T(enum, SHRT_MAX)			\
  T(enum, SHRT_MIN)			\
  T(enum, INT_MAX)			\
  T(enum, INT_MIN)			\
  T(enum, 0x123)			\
  S(wrapstring, (char *) "")		\
  S(wrapstring, (char *) "hello, world")

#define T(type, val) \
  v_##type = val;			\
  if (! xdr_##type (&xdrs, &v_##type))	\
    {					\
      puts ("encoding of " #type	\
	    " " #val " failed");	\
      return 1;				\
    }
#define S(type, val) T(type, val)

  TESTS
#undef T
#undef S

  xdr_destroy (&xdrs);

  xdrmem_create (&xdrs, (char *) buf, sizeof (buf), XDR_DECODE);

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
#define S(type, val) \
  v_##type = NULL;			\
  if (! xdr_##type (&xdrs, &v_##type))	\
    {					\
      puts ("decoding of " #type	\
	    " " #val " failed");	\
      return 1;				\
    }					\
  if (strcmp (v_##type, val))		\
    {					\
      puts ("decoded value differs, "	\
	    "type " #type " " #val);	\
      return 1;				\
    }					\
  free (v_##type);			\
  v_##type = NULL;

  TESTS
#undef T
#undef S

  xdr_destroy (&xdrs);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
