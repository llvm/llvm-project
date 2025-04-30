/* Internal header for netgroup related functions.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _NETGROUP_H
#define _NETGROUP_H	1

#include <nsswitch.h>

/* A netgroup can consist of names of other netgroups.  We have to
   track which netgroups were read and which still have to be read.  */
struct name_list
{
  struct name_list *next;
  char name[];
};


/* Dataset for iterating netgroups.  */
struct __netgrent
{
  enum { triple_val, group_val } type;

  union
  {
    struct
    {
      const char *host;
      const char *user;
      const char *domain;
    }
    triple;

    const char *group;
  } val;

  /* Room for the data kept between the calls to the netgroup
     functions.  We must avoid global variables.  */
  char *data;
  size_t data_size;
  union
  {
    char *cursor;
    unsigned long int position;
  };
  int first;

  struct name_list *known_groups;
  struct name_list *needed_groups;

  /* This handle for the NSS data base is shared between all
     set/get/endXXXent functions.  */
  struct nss_action *nip;
};


/* The internal netgroup handling functions might be called from outside.  */
extern int __internal_setnetgrent (const char *group,
				   struct __netgrent *datap);
libc_hidden_proto (__internal_setnetgrent)
extern void __internal_endnetgrent (struct __netgrent *datap);
libc_hidden_proto (__internal_endnetgrent)
extern int __internal_getnetgrent_r (char **hostp, char **userp,
				     char **domainp, struct __netgrent *datap,
				     char *buffer, size_t buflen, int *errnop);
libc_hidden_proto (__internal_getnetgrent_r)

#endif /* netgroup.h */
