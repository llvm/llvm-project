/* Implementation of the dngettext(3) function.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "gettextP.h"

#include <locale.h>

#ifdef _LIBC
# include <libintl.h>
#else
# include "libgnuintl.h"
#endif

/* @@ end of prolog @@ */

/* Names for the libintl functions are a problem.  They must not clash
   with existing names and they should follow ANSI C.  But this source
   code is also used in GNU C Library where the names have a __
   prefix.  So we have to make a difference here.  */
#ifdef _LIBC
# define DNGETTEXT __dngettext
# define DCNGETTEXT __dcngettext
#else
# define DNGETTEXT libintl_dngettext
# define DCNGETTEXT libintl_dcngettext
#endif

/* Look up MSGID in the DOMAINNAME message catalog of the current
   LC_MESSAGES locale and skip message according to the plural form.  */
char *
DNGETTEXT (const char *domainname,
	   const char *msgid1, const char *msgid2, unsigned long int n)
{
  return DCNGETTEXT (domainname, msgid1, msgid2, n, LC_MESSAGES);
}

#ifdef _LIBC
/* Alias for function name in GNU C Library.  */
weak_alias (__dngettext, dngettext);
#endif
