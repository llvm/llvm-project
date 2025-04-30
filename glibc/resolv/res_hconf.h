/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger (davidm@azstarnet.com).

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

#ifndef _RES_HCONF_H_
#define _RES_HCONF_H_

#include <netdb.h>

#define TRIMDOMAINS_MAX	4

struct hconf
{
  /* We keep the INITIALIZED member only for backwards compatibility.  New
     code should just call _res_hconf_init unconditionally.  For this field
     to be used safely, users must ensure that either (1) a call to
     _res_hconf_init happens-before any load from INITIALIZED, or (2) an
     assignment of zero to INITIALIZED happens-before any load from it, and
     these loads use acquire MO if the intent is to skip calling
     _res_hconf_init if the load returns a nonzero value.  Such acquire MO
     loads will then synchronize with the release MO store to INITIALIZED
     in do_init in res_hconf.c; see pthread_once for more detail.  */
  int initialized;
  int unused1;
  int unused2[4];
  int num_trimdomains;
  const char *trimdomain[TRIMDOMAINS_MAX];
  unsigned int flags;
#  define HCONF_FLAG_INITED	(1 << 0) /* initialized? */
#  define HCONF_FLAG_REORDER	(1 << 3) /* list best address first */
#  define HCONF_FLAG_MULTI	(1 << 4) /* see comments for gethtbyname() */
};
extern struct hconf _res_hconf;

extern void _res_hconf_init (void) attribute_hidden;
extern void _res_hconf_trim_domain (char *domain);
extern void _res_hconf_trim_domains (struct hostent *hp);
extern void _res_hconf_reorder_addrs (struct hostent *hp);

#endif /* _RES_HCONF_H_ */
