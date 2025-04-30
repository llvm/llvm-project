/* Copyright (C) 2008-2021 Free Software Foundation, Inc.
   Contributed by Andreas Krebbel <Andreas.Krebbel@de.ibm.com>.
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

#include <sys/types.h>
#include <utmp.h>
#include <errno.h>
#include <stdlib.h>

#include "utmp32.h"
#include "utmp-convert.h"

/* Allocate a static buffer to be returned to the caller.  As well as
   with the existing version of these functions the caller has to be
   aware that the contents of this buffer will change with subsequent
   calls.  */
#define ALLOCATE_UTMP32_OUT(OUT)			\
  static struct utmp32 *OUT = NULL;			\
							\
  if (OUT == NULL)					\
    {							\
      OUT = malloc (sizeof (struct utmp32));		\
      if (OUT == NULL)					\
	return NULL;					\
    }

/* Perform a lookup for a utmp entry matching FIELD using function
   FUNC.  FIELD is converted to a 64 bit utmp and the result is
   converted back to 32 bit utmp.  */
#define ACCESS_UTMP_ENTRY(FUNC, FIELD)			\
  struct utmp in64;					\
  struct utmp *out64;					\
  ALLOCATE_UTMP32_OUT (out32);				\
							\
  utmp_convert32to64 (FIELD, &in64);			\
  out64 = FUNC (&in64);					\
							\
  if (out64 == NULL)					\
    return NULL;					\
							\
  utmp_convert64to32 (out64, out32);			\
							\
  return out32;

/* Search forward from the current point in the utmp file until the
   next entry with a ut_type matching ID->ut_type.  */
struct utmp32 *
getutid32 (const struct utmp32 *id)
{
  ACCESS_UTMP_ENTRY (__getutid, id)
}
symbol_version (getutid32, getutid, GLIBC_2.0);

/* Search forward from the current point in the utmp file until the
   next entry with a ut_line matching LINE->ut_line.  */
struct utmp32 *
getutline32 (const struct utmp32 *line)
{
  ACCESS_UTMP_ENTRY (__getutline, line)
}
symbol_version (getutline32, getutline, GLIBC_2.0);

/* Write out entry pointed to by UTMP_PTR into the utmp file.  */
struct utmp32 *
pututline32 (const struct utmp32 *utmp_ptr)
{
  ACCESS_UTMP_ENTRY (__pututline, utmp_ptr)
}
symbol_version (pututline32, pututline, GLIBC_2.0);

/* Read next entry from a utmp-like file.  */
struct utmp32 *
getutent32 (void)
{
  struct utmp *out64;
  ALLOCATE_UTMP32_OUT (out32);

  out64 = __getutent ();
  if (!out64)
    return NULL;

  utmp_convert64to32 (out64, out32);
  return out32;
}
symbol_version (getutent32, getutent, GLIBC_2.0);

/* Reentrant versions of the file for handling utmp files.  */

int
getutent32_r (struct utmp32 *buffer, struct utmp32 **result)
{
  struct utmp out64;
  struct utmp *out64p;
  int ret;

  ret = __getutent_r (&out64, &out64p);
  if (ret == -1)
    {
      *result = NULL;
      return -1;
    }

  utmp_convert64to32 (out64p, buffer);
  *result = buffer;

  return 0;
}
symbol_version (getutent32_r, getutent_r, GLIBC_2.0);

int
getutid32_r (const struct utmp32 *id, struct utmp32 *buffer,
	       struct utmp32 **result)
{
  struct utmp in64;
  struct utmp out64;
  struct utmp *out64p;
  int ret;

  utmp_convert32to64 (id, &in64);

  ret = __getutid_r (&in64, &out64, &out64p);
  if (ret == -1)
    {
      *result = NULL;
      return -1;
    }

  utmp_convert64to32 (out64p, buffer);
  *result = buffer;

  return 0;
}
symbol_version (getutid32_r, getutid_r, GLIBC_2.0);

int
getutline32_r (const struct utmp32 *line,
		 struct utmp32 *buffer, struct utmp32 **result)
{
  struct utmp in64;
  struct utmp out64;
  struct utmp *out64p;
  int ret;

  utmp_convert32to64 (line, &in64);

  ret = __getutline_r (&in64, &out64, &out64p);
  if (ret == -1)
    {
      *result = NULL;
      return -1;
    }

  utmp_convert64to32 (out64p, buffer);
  *result = buffer;

  return 0;

}
symbol_version (getutline32_r, getutline_r, GLIBC_2.0);

/* Append entry UTMP to the wtmp-like file WTMP_FILE.  */
void
updwtmp32 (const char *wtmp_file, const struct utmp32 *utmp)
{
  struct utmp in32;

  utmp_convert32to64 (utmp, &in32);
  __updwtmp (wtmp_file, &in32);
}
symbol_version (updwtmp32, updwtmp, GLIBC_2.0);
