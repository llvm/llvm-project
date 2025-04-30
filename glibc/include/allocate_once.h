/* Allocate and initialize an object once, in a thread-safe fashion.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#ifndef _ALLOCATE_ONCE_H
#define _ALLOCATE_ONCE_H

#include <atomic.h>

/* Slow path for allocate_once; see below.  */
void *__libc_allocate_once_slow (void **__place,
                                 void *(*__allocate) (void *__closure),
                                 void (*__deallocate) (void *__closure,
                                                       void *__ptr),
                                 void *__closure);

#ifdef __clang__
/* clang needs the hidden proto to be specified ahead of its use.  */
libc_hidden_proto (__libc_allocate_once_slow)
#endif

/* Return an a pointer to an allocated and initialized data structure.
   If this function returns a non-NULL value, the caller can assume
   that pointed-to data has been initialized according to the ALLOCATE
   function.

   It is expected that callers define an inline helper function which
   adds type safety, like this.

   struct foo { ... };
   struct foo *global_foo;
   static void *allocate_foo (void *closure);
   static void *deallocate_foo (void *closure, void *ptr);

   static inline struct foo *
   get_foo (void)
   {
     return allocate_once (&global_foo, allocate_foo, free_foo, NULL);
   }

   (Note that the global_foo variable is initialized to zero.)
   Usage of this helper function looks like this:

   struct foo *local_foo = get_foo ();
   if (local_foo == NULL)
      report_allocation_failure ();

   allocate_once first performs an acquire MO load on *PLACE.  If the
   result is not null, it is returned.  Otherwise, ALLOCATE (CLOSURE)
   is called, yielding a value RESULT.  If RESULT equals NULL,
   allocate_once returns NULL, and does not modify *PLACE (but another
   thread may concurrently perform an allocation which succeeds,
   updating *PLACE).  If RESULT does not equal NULL, the function uses
   a CAS with acquire-release MO to update the NULL value in *PLACE
   with the RESULT value.  If it turns out that *PLACE was updated
   concurrently, allocate_once calls DEALLOCATE (CLOSURE, RESULT) to
   undo the effect of ALLOCATE, and returns the new value of *PLACE
   (after an acquire MO load).  If DEALLOCATE is NULL, free (RESULT)
   is called instead.

   Compared to __libc_once, allocate_once has the advantage that it
   does not need separate space for a control variable, and that it is
   safe with regards to cancellation and other forms of exception
   handling if the supplied callback functions are safe in that
   regard.  allocate_once passes a closure parameter to the allocation
   function, too.  */
static inline void *
allocate_once (void **__place, void *(*__allocate) (void *__closure),
               void (*__deallocate) (void *__closure, void *__ptr),
               void *__closure)
{
  /* Synchronizes with the release MO CAS in
     __allocate_once_slow.  */
  void *__result = atomic_load_acquire (__place);
  if (__result != NULL)
    return __result;
  else
    return __libc_allocate_once_slow (__place, __allocate, __deallocate,
                                      __closure);
}

#ifndef _ISOMAC
libc_hidden_proto (__libc_allocate_once_slow)
#endif

#endif /* _ALLOCATE_ONCE_H */
