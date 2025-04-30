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

#include <errno.h>
#include <nss.h>
#include <stdio.h>
#include <string.h>
#include <grp.h>

#define flockfile(s) _IO_flockfile (s)
#define funlockfile(s) _IO_funlockfile (s)

#define _S(x)	x ? x : ""

/* Write an entry to the given stream.
   This must know the format of the group file.  */
int
putgrent (const struct group *gr, FILE *stream)
{
  int retval;

  if (__glibc_unlikely (gr == NULL) || __glibc_unlikely (stream == NULL)
      || gr->gr_name == NULL || !__nss_valid_field (gr->gr_name)
      || !__nss_valid_field (gr->gr_passwd)
      || !__nss_valid_list_field (gr->gr_mem))
    {
      __set_errno (EINVAL);
      return -1;
    }

  flockfile (stream);

  if (gr->gr_name[0] == '+' || gr->gr_name[0] == '-')
    retval = fprintf (stream, "%s:%s::",
		      gr->gr_name, _S (gr->gr_passwd));
  else
    retval = fprintf (stream, "%s:%s:%lu:",
		      gr->gr_name, _S (gr->gr_passwd),
		      (unsigned long int) gr->gr_gid);
  if (__builtin_expect (retval, 0) < 0)
    {
      funlockfile (stream);
      return -1;
    }

  if (gr->gr_mem != NULL)
    {
      for (size_t i = 0; gr->gr_mem[i] != NULL; i++)
	if (fprintf (stream, i == 0 ? "%s" : ",%s", gr->gr_mem[i]) < 0)
	  {
	    /* What else can we do?  */
	    funlockfile (stream);
	    return -1;
	  }
    }

  retval = fputc_unlocked ('\n', stream);

  funlockfile (stream);

  return retval < 0 ? -1 : 0;
}
