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

#ifdef SHARED
# include <sys/types.h>
# include <utmp.h>
# include <libc-symbols.h>

# include "utmp32.h"
# include "utmp-convert.h"

/* Write the given entry into utmp and wtmp.  */
void
login32 (const struct utmp32 *entry)
{
  struct utmp in64;

  utmp_convert32to64 (entry, &in64);
  login (&in64);
}

symbol_version (login32, login, GLIBC_2.0);
#endif
