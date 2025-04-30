/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <atomic.h>
#include <ldsodefs.h>
#include <libintl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sysdep.h>
#include <unistd.h>
#include <sys/mman.h>


extern const char *__progname;

#include <wchar.h>
#include <libio/iolibio.h>
#define fflush(s) _IO_fflush (s)

/* This function, when passed a string containing an asserted
   expression, a filename, and a line number, prints a message
   on the standard error stream of the form:
	a.c:10: foobar: Assertion `a == b' failed.
   It then aborts program execution via a call to `abort'.  */

#ifdef FATAL_PREPARE_INCLUDE
# include FATAL_PREPARE_INCLUDE
#endif


void
__assert_fail_base (const char *fmt, const char *assertion, const char *file,
		    unsigned int line, const char *function)
{
  char *str;

#ifdef FATAL_PREPARE
  FATAL_PREPARE;
#endif

  int total;
  if (__asprintf (&str, fmt,
		  __progname, __progname[0] ? ": " : "",
		  file, line,
		  function ? function : "", function ? ": " : "",
		  assertion, &total) >= 0)
    {
      /* Print the message.  */
      (void) __fxprintf (NULL, "%s", str);
      (void) fflush (stderr);

      total = (total + 1 + GLRO(dl_pagesize) - 1) & ~(GLRO(dl_pagesize) - 1);
      struct abort_msg_s *buf = __mmap (NULL, total, PROT_READ | PROT_WRITE,
					MAP_ANON | MAP_PRIVATE, -1, 0);
      if (__glibc_likely (buf != MAP_FAILED))
	{
	  buf->size = total;
	  strcpy (buf->msg, str);

	  /* We have to free the old buffer since the application might
	     catch the SIGABRT signal.  */
	  struct abort_msg_s *old = atomic_exchange_acq (&__abort_msg, buf);

	  if (old != NULL)
	    __munmap (old, old->size);
	}

      free (str);
    }
  else
    {
      /* At least print a minimal message.  */
      static const char errstr[] = "Unexpected error.\n";
      __libc_write (STDERR_FILENO, errstr, sizeof (errstr) - 1);
    }

  abort ();
}


#undef __assert_fail
void
__assert_fail (const char *assertion, const char *file, unsigned int line,
	       const char *function)
{
  __assert_fail_base (_("%s%s%s:%u: %s%sAssertion `%s' failed.\n%n"),
		      assertion, file, line, function);
}
hidden_def(__assert_fail)
