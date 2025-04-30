/* Return error detail for failing <dlfcn.h> functions.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <libintl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libc-lock.h>
#include <ldsodefs.h>
#include <libc-symbols.h>
#include <assert.h>
#include <dlerror.h>

char *
__dlerror (void)
{
# ifdef SHARED
  if (!rtld_active ())
    return GLRO (dl_dlfcn_hook)->dlerror ();
# endif

  struct dl_action_result *result = __libc_dlerror_result;

  /* No libdl function has been called.  No error is possible.  */
  if (result == NULL)
    return NULL;

  /* For an early malloc failure, clear the error flag and return the
     error message.  This marks the error as delivered.  */
  if (result == dl_action_result_malloc_failed)
    {
      __libc_dlerror_result = NULL;
      return (char *) "out of memory";
    }

  /* Placeholder object.  This can be observed in a recursive call,
     e.g. from an ELF constructor.  */
  if (result->errstring == NULL)
    return NULL;

  /* If we have already reported the error, we can free the result and
     return NULL.  See __libc_dlerror_result_free.  */
  if (result->returned)
    {
      __libc_dlerror_result = NULL;
      dl_action_result_errstring_free (result);
      free (result);
      return NULL;
    }

  assert (result->errstring != NULL);

  /* Create the combined error message.  */
  char *buf;
  int n;
  if (result->errcode == 0)
    n = __asprintf (&buf, "%s%s%s",
		    result->objname,
		    result->objname[0] == '\0' ? "" : ": ",
		    _(result->errstring));
  else
    {
      __set_errno (result->errcode);
      n = __asprintf (&buf, "%s%s%s: %m",
		      result->objname,
		      result->objname[0] == '\0' ? "" : ": ",
		      _(result->errstring));
      /* Set errno again in case asprintf clobbered it.  */
      __set_errno (result->errcode);
    }

  /* Mark the error as delivered.  */
  result->returned = true;

  if (n >= 0)
    {
      /* Replace the error string with the newly allocated one.  */
      dl_action_result_errstring_free (result);
      result->errstring = buf;
      result->errstring_source = dl_action_result_errstring_local;
      return buf;
    }
  else
    /* We could not create the combined error message, so use the
       existing string as a fallback.  */
    return result->errstring;
}
versioned_symbol (libc, __dlerror, dlerror, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libdl, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libdl, __dlerror, dlerror, GLIBC_2_0);
#endif

int
_dlerror_run (void (*operate) (void *), void *args)
{
  struct dl_action_result *result = __libc_dlerror_result;
  if (result != NULL)
    {
      if (result == dl_action_result_malloc_failed)
	{
	  /* Clear the previous error.  */
	  __libc_dlerror_result = NULL;
	  result = NULL;
	}
      else
	{
	  /* There is an existing object.  Free its error string, but
	     keep the object.  */
	  dl_action_result_errstring_free (result);
	  /* Mark the object as not containing an error.  This ensures
	     that call to dlerror from, for example, an ELF
	     constructor will not notice this result object.  */
	  result->errstring = NULL;
	}
    }

  const char *objname;
  const char *errstring;
  bool malloced;
  int errcode = GLRO (dl_catch_error) (&objname, &errstring, &malloced,
				       operate, args);

  /* ELF constructors or destructors may have indirectly altered the
     value of __libc_dlerror_result, therefore reload it.  */
  result = __libc_dlerror_result;

  if (errstring == NULL)
    {
      /* There is no error.  We no longer need the result object if it
	 does not contain an error.  However, a recursive call may
	 have added an error even if this call did not cause it.  Keep
	 the other error.  */
      if (result != NULL && result->errstring == NULL)
	{
	  __libc_dlerror_result = NULL;
	  free (result);
	}
      return 0;
    }
  else
    {
      /* A new error occurred.  Check if a result object has to be
	 allocated.  */
      if (result == NULL || result == dl_action_result_malloc_failed)
	{
	  /* Allocating storage for the error message after the fact
	     is not ideal.  But this avoids an infinite recursion in
	     case malloc itself calls libdl functions (without
	     triggering errors).  */
	  result = malloc (sizeof (*result));
	  if (result == NULL)
	    {
	      /* Assume that the dlfcn failure was due to a malloc
		 failure, too.  */
	      if (malloced)
		dl_error_free ((char *) errstring);
	      __libc_dlerror_result = dl_action_result_malloc_failed;
	      return 1;
	    }
	  __libc_dlerror_result = result;
	}
      else
	/* Deallocate the existing error message from a recursive
	   call, but reuse the result object.  */
	dl_action_result_errstring_free (result);

      result->errcode = errcode;
      result->objname = objname;
      result->errstring = (char *) errstring;
      result->returned = false;
      /* In case of an error, the malloced flag indicates whether the
	 error string is constant or not.  */
      if (malloced)
	result->errstring_source = dl_action_result_errstring_rtld;
      else
	result->errstring_source = dl_action_result_errstring_constant;

      return 1;
    }
}
