/* Extended resolver state separate from struct __res_state.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#ifndef RESOLV_STATE_H
#define RESOLV_STATE_H

#include <netinet/in.h>
#include <stdbool.h>
#include <stddef.h>

/* This type corresponds to members of the _res.sort_list array.  */
struct resolv_sortlist_entry
{
  struct in_addr addr;
  uint32_t mask;
};

/* Extended resolver state associated with res_state objects.  Client
   code can reach this state through a struct resolv_context
   object.  */
struct resolv_conf
{
  /* Reference counter.  The object is deallocated once it reaches
     zero.  For internal use within resolv_conf only.  */
  size_t __refcount;

  /* List of IPv4 and IPv6 name server addresses.  */
  const struct sockaddr **nameserver_list;
  size_t nameserver_list_size;

  /* The domain names forming the search list.  */
  const char *const *search_list;
  size_t search_list_size;

  /* IPv4 address preference rules.  */
  const struct resolv_sortlist_entry *sort_list;
  size_t sort_list_size;

  /* _res.options has type unsigned long, but we can only use 32 bits
     for portability across all architectures.  */
  unsigned int options;
  unsigned int retrans;         /* Timeout.  */
  unsigned int retry;           /* Number of times to retry.  */
  unsigned int ndots; /* Dots needed for initial non-search query.  */
};

/* The functions below are for use by the res_init resolv.conf parser
   and the struct resolv_context facility.  */

struct __res_state;
struct file_change_detection;

/* Read /etc/resolv.conf and return a configuration object, or NULL if
   /etc/resolv.conf cannot be read due to memory allocation errors.
   If PREINIT is not NULL, some configuration values are taken from
   the struct __res_state object.  If CHANGE is not null, file change
   detection data is written to *CHANGE, based on the state of the
   file after reading it.  */
struct resolv_conf *__resolv_conf_load (struct __res_state *preinit,
                                        struct file_change_detection *change)
  attribute_hidden __attribute__ ((warn_unused_result));

/* Return a configuration object for the current /etc/resolv.conf
   settings, or NULL on failure.  The object is cached.  */
struct resolv_conf *__resolv_conf_get_current (void)
  attribute_hidden __attribute__ ((warn_unused_result));

/* Return the extended resolver state for *RESP, or NULL if it cannot
   be determined.  A call to this function must be paired with a call
   to __resolv_conf_put.  */
struct resolv_conf *__resolv_conf_get (struct __res_state *) attribute_hidden;

/* Converse of __resolv_conf_get.  */
void __resolv_conf_put (struct resolv_conf *) attribute_hidden;

/* Allocate a new struct resolv_conf object and copy the
   pre-configured values from *INIT.  Return NULL on allocation
   failure.  The object must be deallocated using
   __resolv_conf_put.  */
struct resolv_conf *__resolv_conf_allocate (const struct resolv_conf *init)
  attribute_hidden __attribute__ ((nonnull (1), warn_unused_result));

/* Associate an existing extended resolver state with *RESP.  Return
   false on allocation failure.  In addition, update *RESP with the
   overlapping non-extended resolver state.  */
bool __resolv_conf_attach (struct __res_state *, struct resolv_conf *)
  attribute_hidden;

/* Detach the extended resolver state from *RESP.  */
void __resolv_conf_detach (struct __res_state *resp) attribute_hidden;

#endif /* RESOLV_STATE_H */
