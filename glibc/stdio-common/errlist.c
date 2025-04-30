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

#include <stdio.h>
#include <errno.h>
#include <libintl.h>
#include <array_length.h>

#ifndef ERR_MAP
# define ERR_MAP(n) n
#endif

const char *const _sys_errlist_internal[] =
  {
#define _S(n, str)         [ERR_MAP(n)] = str,
#include <errlist.h>
#undef _S
  };

const char *
__get_errlist (int errnum)
{
  if (errnum >= 0 && errnum < array_length (_sys_errlist_internal))
    return _sys_errlist_internal[errnum];
  return NULL;
}

static const union sys_errname_t
{
  struct
  {
#define MSGSTRFIELD1(line) str##line
#define MSGSTRFIELD(line)  MSGSTRFIELD1(line)
#define _S(n, str)         char MSGSTRFIELD(__LINE__)[sizeof(#n)];
#include <errlist.h>
#undef _S
  };
  char str[0];
} _sys_errname = { {
#define _S(n, s) #n,
#include <errlist.h>
#undef _S
} };

static const unsigned short _sys_errnameidx[] =
{
#define _S(n, s) \
  [ERR_MAP(n)] = offsetof(union sys_errname_t, MSGSTRFIELD(__LINE__)),
#include <errlist.h>
#undef _S
};

const char *
__get_errname (int errnum)
{
  if (errnum < 0 || errnum >= array_length (_sys_errnameidx)
      || (errnum > 0 && _sys_errnameidx[errnum] == 0))
    return NULL;
  return _sys_errname.str + _sys_errnameidx[errnum];
}

#include <errlist-compat.c>
