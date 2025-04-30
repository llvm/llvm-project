/* Support for chains recording users of a resource; `struct hurd_userlink'.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef	_HURD_USERLINK_H

#define	_HURD_USERLINK_H	1
#include <features.h>

#define __need_NULL
#include <stddef.h>

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
#  include <hurd/signal.h>
# endif
#endif
#include <setjmp.h>


/* This structure records a link in two doubly-linked lists.
   We call these the per-resource user list and the per-thread
   active-resource list.

   Users of a given resource are recorded by their presence in a list
   associated with that resource.  A user attaches his own link (in local
   storage on his stack) to a shared chain at the time he begins using some
   resource.  When finished with that resource, the user removes his link
   from the chain.  If his link is the last (there are no other users of
   the resource), and his chain has been detached from the shared cell (the
   resource in the cell has been replaced), then the user deallocates the
   resource that he used.

   All uses of shared resources by a single thread are linked together by
   its `active-resource' list; the head of this list is stored in the
   per-thread sigstate structure.  When the thread makes a non-local exit
   (i.e. longjmp), it will examine its active-resource list, and each link
   residing in a stack frame being jumped out of will be unlinked from both
   the resource's user list and the thread's active-resource list, and
   deallocate the resource if that was the last user link for that resource.

   NOTE: Access to a thread's active-resource list must always be done
   inside a signal-proof critical section; the functions in this file
   assume they are called inside a critical section, and do no locking of
   their own.  Also important: the longjmp cleanup relies on all userlink
   structures residing on the stack of the using thread.  */

struct hurd_userlink
  {
    struct
      {
	struct hurd_userlink *next, **prevp;
      } resource, thread;

    /* This function is called when a non-local exit
       unwinds the frame containing this link.  */
    void (*cleanup) (void *cleanup_data, jmp_buf env, int val);
    void *cleanup_data;
  };


#ifndef _HURD_USERLINK_H_EXTERN_INLINE
#define _HURD_USERLINK_H_EXTERN_INLINE __extern_inline
#endif


/* Attach LINK to the chain of users at *CHAINP.  */

extern void
_hurd_userlink_link (struct hurd_userlink **chainp,
		     struct hurd_userlink *link);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_USERLINK_H_EXTERN_INLINE void
_hurd_userlink_link (struct hurd_userlink **chainp,
		     struct hurd_userlink *link)
{
  struct hurd_userlink **thread_chainp;

  link->resource.next = *chainp;
  if (link->resource.next)
    link->resource.next->resource.prevp = &link->resource.next;
  link->resource.prevp = chainp;
  *chainp = link;

  /* Also chain it on the current thread's list of active resources.  */
  thread_chainp = &_hurd_self_sigstate ()->active_resources;
  link->thread.next = *thread_chainp;
  if (link->thread.next)
    link->thread.next->thread.prevp = &link->thread.next;
  link->thread.prevp = thread_chainp;
  *thread_chainp = link;
}
# endif
#endif


/* Detach LINK from its chain.  Returns nonzero iff this was the
   last user of the resource and it should be deallocated.  */

extern int _hurd_userlink_unlink (struct hurd_userlink *link);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_USERLINK_H_EXTERN_INLINE int
_hurd_userlink_unlink (struct hurd_userlink *link)
{
  /* We should deallocate the resource used if this chain has been detached
     from the cell (and thus has a nil `prevp'), and there is no next link
     representing another user reference to the same resource. */
  int dealloc = ! link->resource.next && ! link->resource.prevp;

  /* Remove our link from the chain of current users.  */
  if (link->resource.prevp)
    *link->resource.prevp = link->resource.next;
  if (link->resource.next)
    link->resource.next->resource.prevp = link->resource.prevp;

  /* Remove our link from the chain of currently active resources
     for this thread.  */
  *link->thread.prevp = link->thread.next;
  if (link->thread.next)
    link->thread.next->thread.prevp = link->thread.prevp;

  return dealloc;
}
# endif
#endif

/* Relocate LINK to NEW_LINK.
   To be used when e.g. reallocating a link array.  */

extern void _hurd_userlink_move (struct hurd_userlink *new_link,
                                struct hurd_userlink *link);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_USERLINK_H_EXTERN_INLINE void
_hurd_userlink_move (struct hurd_userlink *new_link,
                     struct hurd_userlink *link)
{
  *new_link = *link;

  if (new_link->resource.next != NULL)
    new_link->resource.next->resource.prevp = &new_link->resource.next;
  *new_link->resource.prevp = new_link;

  if (new_link->thread.next != NULL)
    new_link->thread.next->thread.prevp = &new_link->thread.next;
  *new_link->thread.prevp = new_link;
}
# endif
#endif

/* Clear all users from *CHAINP.  Call this when the resource *CHAINP
   protects is changing.  If the return value is nonzero, no users are on
   the chain and the caller should deallocate the resource.  If the return
   value is zero, someone is still using the resource and they will
   deallocate it when they are finished.  */

extern int _hurd_userlink_clear (struct hurd_userlink **chainp);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_USERLINK_H_EXTERN_INLINE int
_hurd_userlink_clear (struct hurd_userlink **chainp)
{
  if (*chainp == NULL)
    return 1;

  /* Detach the chain of current users from the cell.  The last user to
     remove his link from that chain will deallocate the old resource.  */
  (*chainp)->resource.prevp = NULL;
  *chainp = NULL;
  return 0;
}
# endif
#endif

#endif	/* hurd/userlink.h */
