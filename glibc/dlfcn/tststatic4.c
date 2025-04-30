/* Global object symbol access tests with a static executable (BZ #15022).
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

#define MAGIC0 0
#define MAGIC1 0x5500ffaa
#define MAGIC2 0xaaff0055
#define MAGIC3 0xff55aa00

/* Check the ability to access the global symbol object and then
   global-scope symbol access consistency via different mappings
   requested from a static executable.  */
static int
do_test (void)
{
  unsigned int (*initial_getfoo) (void);
  void (*initial_setfoo) (unsigned int);
  unsigned int (*global_getfoo) (void);
  void (*global_setfoo) (unsigned int);
  unsigned int (*local_getfoo) (void);
  void (*local_setfoo) (unsigned int);
  unsigned int *initial_foop;
  unsigned int *global_foop;
  unsigned int *local_foop;
  void *initial_handle;
  void *global_handle;
  void *local_handle;
  unsigned int foo;

  /* Try to map self.  */
  initial_handle = dlopen (NULL, RTLD_LAZY | RTLD_GLOBAL);
  if (initial_handle == NULL)
    {
      printf ("dlopen [initial] (NULL): %s\n", dlerror ());
      return 1;
    }

  /* Make sure symbol lookups fail gracefully.  */
  initial_foop = dlsym (initial_handle, "foo");
  if (initial_foop != NULL)
    {
      printf ("dlsym [initial] (foo): got %p, expected NULL\n", initial_foop);
      return 1;
    }

  initial_getfoo = dlsym (initial_handle, "getfoo");
  if (initial_getfoo != NULL)
    {
      printf ("dlsym [initial] (getfoo): got %p, expected NULL\n",
	      initial_getfoo);
      return 1;
    }

  initial_setfoo = dlsym (initial_handle, "setfoo");
  if (initial_setfoo != NULL)
    {
      printf ("dlsym [initial] (setfoo): got %p, expected NULL\n",
	      initial_setfoo);
      return 1;
    }

  /* Try to map a module into the global scope.  */
  global_handle = dlopen ("modstatic3.so", RTLD_LAZY | RTLD_GLOBAL);
  if (global_handle == NULL)
    {
      printf ("dlopen [global] (modstatic3.so): %s\n", dlerror ());
      return 1;
    }

  /* Get at its symbols.  */
  global_foop = dlsym (global_handle, "foo");
  if (global_foop == NULL)
    {
      printf ("dlsym [global] (foo): %s\n", dlerror ());
      return 1;
    }

  global_getfoo = dlsym (global_handle, "getfoo");
  if (global_getfoo == NULL)
    {
      printf ("dlsym [global] (getfoo): %s\n", dlerror ());
      return 1;
    }

  global_setfoo = dlsym (global_handle, "setfoo");
  if (global_setfoo == NULL)
    {
      printf ("dlsym [global] (setfoo): %s\n", dlerror ());
      return 1;
    }

  /* Try to map self again now.  */
  local_handle = dlopen (NULL, RTLD_LAZY | RTLD_LOCAL);
  if (local_handle == NULL)
    {
      printf ("dlopen [local] (NULL): %s\n", dlerror ());
      return 1;
    }

  /* Make sure we can get at the previously loaded module's symbols
     via this handle too.  */
  local_foop = dlsym (local_handle, "foo");
  if (local_foop == NULL)
    {
      printf ("dlsym [local] (foo): %s\n", dlerror ());
      return 1;
    }

  local_getfoo = dlsym (local_handle, "getfoo");
  if (local_getfoo == NULL)
    {
      printf ("dlsym [local] (getfoo): %s\n", dlerror ());
      return 1;
    }

  local_setfoo = dlsym (local_handle, "setfoo");
  if (local_setfoo == NULL)
    {
      printf ("dlsym [local] (setfoo): %s\n", dlerror ());
      return 1;
    }

  /* Make sure we can get at the previously loaded module's symbols
     via a handle that was obtained before the module was loaded too.  */
  initial_foop = dlsym (initial_handle, "foo");
  if (initial_foop == NULL)
    {
      printf ("dlsym [initial] (foo): %s\n", dlerror ());
      return 1;
    }

  initial_getfoo = dlsym (initial_handle, "getfoo");
  if (initial_getfoo == NULL)
    {
      printf ("dlsym [initial] (getfoo): %s\n", dlerror ());
      return 1;
    }

  initial_setfoo = dlsym (initial_handle, "setfoo");
  if (initial_setfoo == NULL)
    {
      printf ("dlsym [initial] (setfoo): %s\n", dlerror ());
      return 1;
    }

  /* Make sure the view of the initial state is consistent.  */
  foo = *initial_foop;
  if (foo != MAGIC0)
    {
      printf ("*foop [initial]: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  foo = *global_foop;
  if (foo != MAGIC0)
    {
      printf ("*foop [global]: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  foo = *local_foop;
  if (foo != MAGIC0)
    {
      printf ("*foop [local]: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  foo = initial_getfoo ();
  if (foo != MAGIC0)
    {
      printf ("getfoo [initial]: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  foo = global_getfoo ();
  if (foo != MAGIC0)
    {
      printf ("getfoo [global]: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  foo = local_getfoo ();
  if (foo != MAGIC0)
    {
      printf ("getfoo [local]: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  /* Likewise with a change to its state made through the first handle.  */
  initial_setfoo (MAGIC1);

  foo = *initial_foop;
  if (foo != MAGIC1)
    {
      printf ("*foop [initial]: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  foo = *global_foop;
  if (foo != MAGIC1)
    {
      printf ("*foop [global]: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  foo = *local_foop;
  if (foo != MAGIC1)
    {
      printf ("*foop [local]: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  foo = initial_getfoo ();
  if (foo != MAGIC1)
    {
      printf ("getfoo [initial]: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  foo = global_getfoo ();
  if (foo != MAGIC1)
    {
      printf ("getfoo [global]: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  foo = local_getfoo ();
  if (foo != MAGIC1)
    {
      printf ("getfoo [local]: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  /* Likewise with a change to its state made through the second handle.  */
  global_setfoo (MAGIC2);

  foo = *initial_foop;
  if (foo != MAGIC2)
    {
      printf ("*foop [initial]: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  foo = *global_foop;
  if (foo != MAGIC2)
    {
      printf ("*foop [global]: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  foo = *local_foop;
  if (foo != MAGIC2)
    {
      printf ("*foop [local]: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  foo = initial_getfoo ();
  if (foo != MAGIC2)
    {
      printf ("getfoo [initial]: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  foo = global_getfoo ();
  if (foo != MAGIC2)
    {
      printf ("getfoo [global]: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  foo = local_getfoo ();
  if (foo != MAGIC2)
    {
      printf ("getfoo [local]: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  /* Likewise with a change to its state made through the third handle.  */
  local_setfoo (MAGIC3);

  foo = *initial_foop;
  if (foo != MAGIC3)
    {
      printf ("*foop [initial]: got %#x, expected %#x\n", foo, MAGIC3);
      return 1;
    }

  foo = *global_foop;
  if (foo != MAGIC3)
    {
      printf ("*foop [global]: got %#x, expected %#x\n", foo, MAGIC3);
      return 1;
    }

  foo = *local_foop;
  if (foo != MAGIC3)
    {
      printf ("*foop [local]: got %#x, expected %#x\n", foo, MAGIC3);
      return 1;
    }

  foo = initial_getfoo ();
  if (foo != MAGIC3)
    {
      printf ("getfoo [initial]: got %#x, expected %#x\n", foo, MAGIC3);
      return 1;
    }

  foo = global_getfoo ();
  if (foo != MAGIC3)
    {
      printf ("getfoo [global]: got %#x, expected %#x\n", foo, MAGIC3);
      return 1;
    }

  foo = local_getfoo ();
  if (foo != MAGIC3)
    {
      printf ("getfoo [local]: got %#x, expected %#x\n", foo, MAGIC3);
      return 1;
    }

  /* All done, clean up.  */
  initial_getfoo = NULL;
  initial_setfoo = NULL;
  initial_foop = NULL;

  local_getfoo = NULL;
  local_setfoo = NULL;
  local_foop = NULL;
  dlclose (local_handle);

  global_getfoo = NULL;
  global_setfoo = NULL;
  global_foop = NULL;
  dlclose (global_handle);

  dlclose (initial_handle);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
