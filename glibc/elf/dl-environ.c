/* Environment handling for dynamic loader.
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

#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <ldsodefs.h>

/* Walk through the environment of the process and return all entries
   starting with `LD_'.  */
char *
_dl_next_ld_env_entry (char ***position)
{
  char **current = *position;
  char *result = NULL;

  while (*current != NULL)
    {
      if (__builtin_expect ((*current)[0] == 'L', 0)
	  && (*current)[1] == 'D' && (*current)[2] == '_')
	{
	  result = &(*current)[3];

	  /* Save current position for next visit.  */
	  *position = ++current;

	  break;
	}

      ++current;
    }

  return result;
}


/* In ld.so __environ is not exported.  */
extern char **__environ attribute_hidden;

int
unsetenv (const char *name)
{
  char **ep;

  ep = __environ;
  while (*ep != NULL)
    {
      size_t cnt = 0;

      while ((*ep)[cnt] == name[cnt] && name[cnt] != '\0')
	++cnt;

      if (name[cnt] == '\0' && (*ep)[cnt] == '=')
	{
	  /* Found it.  Remove this pointer by moving later ones to
	     the front.  */
	  char **dp = ep;

	  do
	    dp[0] = dp[1];
	  while (*dp++);
	  /* Continue the loop in case NAME appears again.  */
	}
      else
	++ep;
    }

  return 0;
}
