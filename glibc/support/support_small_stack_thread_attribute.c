/* A thread attribute with a small stack.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <support/xthread.h>
#include <allocate_once.h>

static void *
allocate (void *closure)
{
  pthread_attr_t *result = malloc (sizeof (*result));
  xpthread_attr_init (result);
  support_set_small_thread_stack_size (result);
  return result;
}

static void
deallocate (void *ptr, void *closure)
{
  xpthread_attr_destroy (ptr);
  free (ptr);
}

static void *small_stack_attr;

pthread_attr_t *
support_small_stack_thread_attribute (void)
{
  return allocate_once (&small_stack_attr, allocate, deallocate, NULL);
}

static void __attribute__ ((destructor))
fini (void)
{
  if (small_stack_attr != NULL)
    {
      deallocate (small_stack_attr, NULL);
      small_stack_attr = NULL;
    }
}
