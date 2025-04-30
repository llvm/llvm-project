/* Handle loading/unloading of shared object for transformation.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <assert.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <search.h>
#include <stdlib.h>
#include <string.h>
#include <libc-lock.h>
#include <sys/param.h>

#include <gconv_int.h>
#include <sysdep.h>


#ifdef DEBUG
/* For debugging purposes.  */
static void print_all (void);
#endif


/* This is a tuning parameter.  If a transformation module is not used
   anymore it gets not immediately unloaded.  Instead we wait a certain
   number of load attempts for further modules.  If none of the
   subsequent load attempts name the same object it finally gets unloaded.
   Otherwise it is still available which hopefully is the frequent case.
   The following number is the number of unloading attempts we wait
   before unloading.  */
#define TRIES_BEFORE_UNLOAD	2

/* Array of loaded objects.  This is shared by all threads so we have
   to use semaphores to access it.  */
static void *loaded;

/* Comparison function for searching `loaded_object' tree.  */
static int
known_compare (const void *p1, const void *p2)
{
  const struct __gconv_loaded_object *s1 =
    (const struct __gconv_loaded_object *) p1;
  const struct __gconv_loaded_object *s2 =
    (const struct __gconv_loaded_object *) p2;

  return strcmp (s1->name, s2->name);
}

/* Open the gconv database if necessary.  A non-negative return value
   means success.  */
struct __gconv_loaded_object *
__gconv_find_shlib (const char *name)
{
  struct __gconv_loaded_object *found;
  void *keyp;

  /* Search the tree of shared objects previously requested.  Data in
     the tree are `loaded_object' structures, whose first member is a
     `const char *', the lookup key.  The search returns a pointer to
     the tree node structure; the first member of the is a pointer to
     our structure (i.e. what will be a `loaded_object'); since the
     first member of that is the lookup key string, &FCT_NAME is close
     enough to a pointer to our structure to use as a lookup key that
     will be passed to `known_compare' (above).  */

  keyp = __tfind (&name, &loaded, known_compare);
  if (keyp == NULL)
    {
      /* This name was not known before.  */
      size_t namelen = strlen (name) + 1;

      found = malloc (sizeof (struct __gconv_loaded_object) + namelen);
      if (found != NULL)
	{
	  /* Point the tree node at this new structure.  */
	  found->name = (char *) memcpy (found + 1, name, namelen);
	  found->counter = -TRIES_BEFORE_UNLOAD - 1;
	  found->handle = NULL;

	  if (__builtin_expect (__tsearch (found, &loaded, known_compare)
				== NULL, 0))
	    {
	      /* Something went wrong while inserting the entry.  */
	      free (found);
	      found = NULL;
	    }
	}
    }
  else
    found = *(struct __gconv_loaded_object **) keyp;

  /* Try to load the shared object if the usage count is 0.  This
     implies that if the shared object is not loadable, the handle is
     NULL and the usage count > 0.  */
  if (found != NULL)
    {
      if (found->counter < -TRIES_BEFORE_UNLOAD)
	{
	  assert (found->handle == NULL);
	  found->handle = __libc_dlopen (found->name);
	  if (found->handle != NULL)
	    {
	      found->fct = __libc_dlsym (found->handle, "gconv");
	      if (found->fct == NULL)
		{
		  /* Argh, no conversion function.  There is something
                     wrong here.  */
		  __gconv_release_shlib (found);
		  found = NULL;
		}
	      else
		{
		  found->init_fct = __libc_dlsym (found->handle, "gconv_init");
		  found->end_fct = __libc_dlsym (found->handle, "gconv_end");

#ifdef PTR_MANGLE
		  PTR_MANGLE (found->fct);
		  PTR_MANGLE (found->init_fct);
		  PTR_MANGLE (found->end_fct);
#endif

		  /* We have succeeded in loading the shared object.  */
		  found->counter = 1;
		}
	    }
	  else
	    /* Error while loading the shared object.  */
	    found = NULL;
	}
      else if (found->handle != NULL)
	found->counter = MAX (found->counter + 1, 1);
    }

  return found;
}

static void
do_release_shlib (const void *nodep, VISIT value, void *closure)
{
  struct __gconv_loaded_object *release_handle = closure;
  struct __gconv_loaded_object *obj = *(struct __gconv_loaded_object **) nodep;

  if (value != preorder && value != leaf)
    return;

  if (obj == release_handle)
    {
      /* This is the object we want to unload.  Now decrement the
	 reference counter.  */
      assert (obj->counter > 0);
      --obj->counter;
    }
  else if (obj->counter <= 0 && obj->counter >= -TRIES_BEFORE_UNLOAD
	   && --obj->counter < -TRIES_BEFORE_UNLOAD && obj->handle != NULL)
    {
      /* Unload the shared object.  */
      __libc_dlclose (obj->handle);
      obj->handle = NULL;
    }
}


/* Notify system that a shared object is not longer needed.  */
void
__gconv_release_shlib (struct __gconv_loaded_object *handle)
{
  /* Process all entries.  Please note that we also visit entries
     with release counts <= 0.  This way we can finally unload them
     if necessary.  */
  __twalk_r (loaded, do_release_shlib, handle);
}


/* We run this if we debug the memory allocation.  */
static void __libc_freeres_fn_section
do_release_all (void *nodep)
{
  struct __gconv_loaded_object *obj = (struct __gconv_loaded_object *) nodep;

  /* Unload the shared object.  */
  if (obj->handle != NULL)
    __libc_dlclose (obj->handle);

  free (obj);
}

libc_freeres_fn (free_mem)
{
  __tdestroy (loaded, do_release_all);
  loaded = NULL;
}


#ifdef DEBUG

#include <stdio.h>

static void
do_print (const void *nodep, VISIT value, int level)
{
  struct __gconv_loaded_object *obj = *(struct __gconv_loaded_object **) nodep;

  printf ("%10s: \"%s\", %d\n",
	  value == leaf ? "leaf"
	  : value == preorder ? "preorder"
	  : value == postorder ? "postorder" : "endorder",
	  obj->name, obj->counter);
}

static void __attribute__ ((used))
print_all (void)
{
  __twalk (loaded, do_print);
}
#endif
