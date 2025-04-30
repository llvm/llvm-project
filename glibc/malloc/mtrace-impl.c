/* mtrace implementation for `malloc'.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
                 Written April 2, 1991 by John Gilmore of Cygnus Support.
                 Based on mcheck.c by Mike Haertel.

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


#include <malloc.h>
#include <mcheck.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#include <libc-internal.h>
#include <dso_handle.h>

#include <kernel-features.h>

#define TRACE_BUFFER_SIZE 512

static FILE *mallstream;
static const char mallenv[] = "MALLOC_TRACE";
static char *malloc_trace_buffer;

static void
tr_where (const void *caller, Dl_info *info)
{
  if (caller != NULL)
    {
      if (info != NULL)
        {
          char *buf = (char *) "";
          if (info->dli_sname != NULL)
            {
              size_t len = strlen (info->dli_sname);
              buf = alloca (len + 6 + 2 * sizeof (void *));
	      char sign;
	      ptrdiff_t offset =
		(ptrdiff_t) info->dli_saddr - (ptrdiff_t) caller;

	      if (caller >= (const void *) info->dli_saddr)
		{
		  sign = '+';
		  offset = -offset;
		}
	      else
		  sign = '-';

	      sprintf (buf, "(%s%c%" PRIxPTR ")", info->dli_sname, sign,
		       offset);
            }

	  fprintf (mallstream, "@ %s%s%s[%p] ", info->dli_fname ? : "",
		   info->dli_fname ? ":" : "",
                   buf, caller);
        }
      else
        fprintf (mallstream, "@ [%p] ", caller);
    }
}

static Dl_info *
lock_and_info (const void *caller, Dl_info *mem)
{
  if (caller == NULL)
    return NULL;

  Dl_info *res = dladdr (caller, mem) ? mem : NULL;

  flockfile (mallstream);

  return res;
}

static void
free_mtrace (void *ptr, const void *caller)
{
  if (ptr == NULL)
    return;

  Dl_info mem;
  Dl_info *info = lock_and_info (caller, &mem);
  tr_where (caller, info);
  /* Be sure to print it first.  */
  fprintf (mallstream, "- %p\n", ptr);
  funlockfile (mallstream);
}

static void
malloc_mtrace_after (void *block, size_t size, const void *caller)
{
  Dl_info mem;
  Dl_info *info = lock_and_info (caller, &mem);

  tr_where (caller, info);
  /* We could be printing a NULL here; that's OK.  */
  fprintf (mallstream, "+ %p %#lx\n", block, (unsigned long int) size);

  funlockfile (mallstream);
}

static void
realloc_mtrace_after (void *block, const void *oldptr, size_t size,
		      const void *caller)
{
  Dl_info mem;
  Dl_info *info = lock_and_info (caller, &mem);

  tr_where (caller, info);
  if (block == NULL)
    {
      if (size != 0)
        /* Failed realloc.  */
	fprintf (mallstream, "! %p %#lx\n", oldptr, (unsigned long int) size);
      else
        fprintf (mallstream, "- %p\n", oldptr);
    }
  else if (oldptr == NULL)
    fprintf (mallstream, "+ %p %#lx\n", block, (unsigned long int) size);
  else
    {
      fprintf (mallstream, "< %p\n", oldptr);
      tr_where (caller, info);
      fprintf (mallstream, "> %p %#lx\n", block, (unsigned long int) size);
    }

  funlockfile (mallstream);
}

static void
memalign_mtrace_after (void *block, size_t size, const void *caller)
{
  Dl_info mem;
  Dl_info *info = lock_and_info (caller, &mem);

  tr_where (caller, info);
  /* We could be printing a NULL here; that's OK.  */
  fprintf (mallstream, "+ %p %#lx\n", block, (unsigned long int) size);

  funlockfile (mallstream);
}

/* This function gets called to make sure all memory the library
   allocates get freed and so does not irritate the user when studying
   the mtrace output.  */
static void
release_libc_mem (void)
{
  /* Only call the free function if we still are running in mtrace mode.  */
  if (mallstream != NULL)
    __libc_freeres ();
}

/* We enable tracing if the environment variable MALLOC_TRACE is set.  */

static void
do_mtrace (void)
{
  static int added_atexit_handler;
  char *mallfile;

  /* Don't panic if we're called more than once.  */
  if (mallstream != NULL)
    return;

  mallfile = secure_getenv (mallenv);
  if (mallfile != NULL)
    {
      char *mtb = malloc (TRACE_BUFFER_SIZE);
      if (mtb == NULL)
        return;

      mallstream = fopen (mallfile != NULL ? mallfile : "/dev/null", "wce");
      if (mallstream != NULL)
        {
          /* Be sure it doesn't malloc its buffer!  */
          malloc_trace_buffer = mtb;
          setvbuf (mallstream, malloc_trace_buffer, _IOFBF, TRACE_BUFFER_SIZE);
          fprintf (mallstream, "= Start\n");
          if (!added_atexit_handler)
            {
              added_atexit_handler = 1;
              __cxa_atexit ((void (*)(void *))release_libc_mem, NULL,
			    __dso_handle);
            }
	  __malloc_debug_enable (MALLOC_MTRACE_HOOK);
        }
      else
        free (mtb);
    }
}

static void
do_muntrace (void)
{
  __malloc_debug_disable (MALLOC_MTRACE_HOOK);
  if (mallstream == NULL)
    return;

  /* Do the reverse of what done in mtrace: first reset the hooks and
     MALLSTREAM, and only after that write the trailer and close the
     file.  */
  FILE *f = mallstream;
  mallstream = NULL;

  fprintf (f, "= End\n");
  fclose (f);
}
