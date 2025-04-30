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

#include "utmpx32.h"
#include "utmpx-convert.h"

/* Allocate a static buffer to be returned to the caller.  As well as
   with the existing version of these functions the caller has to be
   aware that the contents of this buffer will change with subsequent
   calls.  */
#define ALLOCATE_UTMPX32_OUT(OUT)			\
  static struct utmpx32 *OUT = NULL;			\
							\
  if (OUT == NULL)					\
    {							\
      OUT = malloc (sizeof (struct utmpx32));		\
      if (OUT == NULL)					\
	return NULL;					\
    }

/* Perform a lookup for a utmpx entry matching FIELD using function
   FUNC.  FIELD is converted to a 64 bit utmpx and the result is
   converted back to 32 bit utmpx.  */
#define ACCESS_UTMPX_ENTRY(FUNC, FIELD)			\
  struct utmpx in64;					\
  struct utmpx *out64;					\
  ALLOCATE_UTMPX32_OUT (out32);				\
							\
  utmpx_convert32to64 (FIELD, &in64);			\
  out64 = FUNC (&in64);					\
							\
  if (out64 == NULL)					\
    return NULL;					\
							\
  utmpx_convert64to32 (out64, out32);			\
							\
  return out32;


/* Get the next entry from the user accounting database.  */
struct utmpx32 *
getutxent32 (void)
{
  struct utmpx *out64;
  ALLOCATE_UTMPX32_OUT (out32);

  out64 = __getutxent ();
  if (!out64)
    return NULL;

  utmpx_convert64to32 (out64, out32);
  return out32;

}
symbol_version (getutxent32, getutxent, GLIBC_2.1);

/* Get the user accounting database entry corresponding to ID.  */
struct utmpx32 *
getutxid32 (const struct utmpx32 *id)
{
  ACCESS_UTMPX_ENTRY (__getutxid, id);
}
symbol_version (getutxid32, getutxid, GLIBC_2.1);

/* Get the user accounting database entry corresponding to LINE.  */
struct utmpx32 *
getutxline32 (const struct utmpx32 *line)
{
  ACCESS_UTMPX_ENTRY (__getutxline, line);
}
symbol_version (getutxline32, getutxline, GLIBC_2.1);

/* Write the entry UTMPX into the user accounting database.  */
struct utmpx32 *
pututxline32 (const struct utmpx32 *utmpx)
{
  ACCESS_UTMPX_ENTRY (__pututxline, utmpx);
}
symbol_version (pututxline32, pututxline, GLIBC_2.1);

/* Append entry UTMP to the wtmpx-like file WTMPX_FILE.  */
void
updwtmpx32 (const char *wtmpx_file, const struct utmpx32 *utmpx)
{
  struct utmpx in64;

  utmpx_convert32to64 (utmpx, &in64);
  __updwtmpx (wtmpx_file, &in64);
}
symbol_version (updwtmpx32, updwtmpx, GLIBC_2.1);

/* Copy the information in UTMPX to UTMP.  */
void
getutmp32 (const struct utmpx32 *utmpx, struct utmp32 *utmp)
{
  struct utmpx in64;
  struct utmp out64;

  utmpx_convert32to64 (utmpx, &in64);
  __getutmp (&in64, &out64);
  utmp_convert64to32 (&out64, utmp);
}
symbol_version (getutmp32, getutmp, GLIBC_2.1.1);

/* Copy the information in UTMP to UTMPX.  */
void
getutmpx32 (const struct utmp32 *utmp, struct utmpx32 *utmpx)
{
  struct utmp in64;
  struct utmpx out64;

  utmp_convert32to64 (utmp, &in64);
  __getutmpx (&in64, &out64);
  utmpx_convert64to32 (&out64, utmpx);
}
symbol_version (getutmpx32, getutmpx, GLIBC_2.1.1);
