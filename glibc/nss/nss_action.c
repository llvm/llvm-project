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

#include <nsswitch.h>

#include <string.h>
#include <libc-lock.h>

/* Maintain a global list of NSS action lists.  Since most databases
   use the same list of actions, this list is usually short.
   Deduplication in __nss_action_allocate ensures that the list does
   not grow without bounds.  */

struct nss_action_list_wrapper
{
  /* The next element of the list.  */
  struct nss_action_list_wrapper *next;

  /* Number of elements in the list (excluding the terminator).  */
  size_t count;

  /* NULL-terminated list of actions.  */
  struct nss_action actions[];
};

/* Toplevel list of allocated NSS action lists.  */
static struct nss_action_list_wrapper *nss_actions;

/* Lock covers the nss_actions list.  */
__libc_lock_define (static, nss_actions_lock);

/* Returns true if the actions are equal (same module, same actions
   array).  */
static bool
actions_equal (const struct nss_action *a, const struct nss_action *b)
{
  return a->module == b->module && a->action_bits == b->action_bits;
}


/* Returns true if COUNT actions at A and B are equal (according to
   actions_equal above). Caller must ensure that either A or B have at
   least COUNT actions.  */
static bool
action_lists_equal (const struct nss_action *a, const struct nss_action *b,
                    size_t count)
{
  for (size_t i = 0; i < count; ++i)
    if (!actions_equal (a + i, b + i))
      return false;
  return true;
}

/* Returns a pre-allocated action list for COUNT actions at ACTIONS,
   or NULL if no such list exists.  */
static nss_action_list
find_allocated (struct nss_action *actions, size_t count)
{
  for (struct nss_action_list_wrapper *p = nss_actions; p != NULL; p = p->next)
    if (p->count == count && action_lists_equal (p->actions, actions, count))
      return p->actions;
  return NULL;
}

nss_action_list
__nss_action_allocate (struct nss_action *actions, size_t count)
{
  nss_action_list result = NULL;
  __libc_lock_lock (nss_actions_lock);

  result = find_allocated (actions, count);
  if (result == NULL)
    {
      struct nss_action_list_wrapper *wrapper
        = malloc (sizeof (*wrapper) + sizeof (*actions) * count);
      if (wrapper != NULL)
        {
          wrapper->next = nss_actions;
          wrapper->count = count;
          memcpy (wrapper->actions, actions, sizeof (*actions) * count);
          nss_actions = wrapper;
          result = wrapper->actions;
        }
    }

  __libc_lock_unlock (nss_actions_lock);
  return result;
}

void __libc_freeres_fn_section
__nss_action_freeres (void)
{
  struct nss_action_list_wrapper *current = nss_actions;
  while (current != NULL)
    {
      struct nss_action_list_wrapper *next = current->next;
      free (current);
      current = next;
    }
  nss_actions = NULL;
}
