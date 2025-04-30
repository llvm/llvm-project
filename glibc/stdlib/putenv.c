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

#if defined _AIX && !defined __GNUC__
 #pragma alloca
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if _LIBC || HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if _LIBC || HAVE_STRING_H
# include <string.h>
#endif

#if !__GNU_LIBRARY__ && !HAVE_STRCHR
# define strchr index
#endif

#ifndef _LIBC
# ifdef HAVE_ALLOCA_H
#  include <alloca.h>
# else
#  ifdef __GNUC__
#   define alloca __builtin_alloca
#  else
extern char *alloca ();
#  endif /* __GNUC__ */
# endif /* HAVE_ALLOCA_H */
#endif /* _LIBC */


/* Put STRING, which is of the form "NAME=VALUE", in the environment.  */
int
putenv (char *string)
{
  const char *const name_end = strchr (string, '=');

  if (name_end != NULL)
    {
      char *name;
#ifdef _LIBC
      int use_malloc = !__libc_use_alloca (name_end - string + 1);
      if (__builtin_expect (use_malloc, 0))
	{
	  name = __strndup (string, name_end - string);
	  if (name == NULL)
	    return -1;
	}
      else
	name = strndupa (string, name_end - string);
#else
# define use_malloc 1
      name = malloc (name_end - string + 1);
      if (name == NULL)
	return -1;
      memcpy (name, string, name_end - string);
      name[name_end - string] = '\0';
#endif
      int result = __add_to_environ (name, NULL, string, 1);

      if (__glibc_unlikely (use_malloc))
	free (name);

      return result;
    }

  __unsetenv (string);
  return 0;
}
