/* NSS actions, elements in a nsswitch.conf configuration line.
   Copyright (c) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _NSS_ACTION_H
#define _NSS_ACTION_H

#include <stddef.h>

/* See nss_database.h for a summary of how this relates.  */

#include "nsswitch.h" /* For lookup_actions.  */

struct nss_module;

/* A NSS action pairs a service module with the action for each result
   state.  */
struct nss_action
{
  /* The service module that provides the functionality (potentially
     not yet loaded).  */
  struct nss_module *module;

  /* Action according to result.  Two bits for each lookup_actions
     value (from nsswitch.h), indexed by enum nss_status (from nss.h).  */
  unsigned int action_bits;
};

/* Value to add to first nss_status value to get zero.  */
#define NSS_STATUS_BIAS 2
/* Number of bits per lookup action.  */
#define NSS_BPL 2
#define NSS_BPL_MASK ((1 << NSS_BPL) - 1)

/* Index in actions of an NSS status.  Note that in nss/nss.h the
   status starts at -2, and we shift that up to zero by adding 2.
   Thus for example NSS_STATUS_TRYAGAIN, which is -2, would index into
   the 0th bit place as expected.  */
static inline int
nss_actions_bits_index (enum nss_status status)
{
  return NSS_BPL * (NSS_STATUS_BIAS + status);
}

/* Returns the lookup_action value for STATUS in ACTION.  */
static inline lookup_actions
nss_action_get (const struct nss_action *action, enum nss_status status)
{
  return ((action->action_bits >> nss_actions_bits_index (status))
	  & NSS_BPL_MASK);
}

/* Sets the lookup_action value for STATUS in ACTION.  */
static inline void
nss_action_set (struct nss_action *action,
                enum nss_status status, lookup_actions actions)
{
  int offset = nss_actions_bits_index (status);
  unsigned int mask = NSS_BPL_MASK << offset;
  action->action_bits = ((action->action_bits & ~mask)
                         | ((unsigned int) actions << offset));
}

static inline void
nss_action_set_all (struct nss_action *action, lookup_actions actions)
{
  unsigned int bits = actions & NSS_BPL_MASK;
  action->action_bits = (   bits
			 | (bits << (NSS_BPL * 1))
			 | (bits << (NSS_BPL * 2))
			 | (bits << (NSS_BPL * 3))
			 | (bits << (NSS_BPL * 4))
			 );
}

/* A list of struct nss_action objects in array terminated by an
   action with a NULL module.  */
typedef struct nss_action *nss_action_list;

/* Returns a pointer to an allocated NSS action list that has COUNT
   actions that matches the array at ACTIONS.  */
nss_action_list __nss_action_allocate (struct nss_action *actions,
                                       size_t count) attribute_hidden;

/* Returns a pointer to a list allocated by __nss_action_allocate, or
   NULL on error.  ENOMEM means a (temporary) memory allocation error,
   EINVAL means that LINE is syntactically invalid.  */
nss_action_list __nss_action_parse (const char *line);

/* Called from __libc_freeres.  */
void __nss_action_freeres (void) attribute_hidden;


#endif /* _NSS_ACTION_H */
