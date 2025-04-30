/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <libintl.h>
#include <string.h>


/* This function, when passed an error number, a filename, and a line
   number, prints a message on the standard error stream of the form:
	a.c:10: foobar: Unexpected error: Computer bought the farm
   It then aborts program execution via a call to `abort'.  */
void
__assert_perror_fail (int errnum,
		      const char *file, unsigned int line,
		      const char *function)
{
  char errbuf[1024];

  char *e = __strerror_r (errnum, errbuf, sizeof errbuf);
  __assert_fail_base (_("%s%s%s:%u: %s%sUnexpected error: %s.\n%n"),
		      e, file, line, function);
}
libc_hidden_def (__assert_perror_fail)
