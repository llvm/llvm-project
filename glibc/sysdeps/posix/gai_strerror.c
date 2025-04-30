/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <pjb27@cam.ac.uk>, 1997.

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

#include <libintl.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>


#define MSGSTRFIELD(line) MSGSTRFIELD1 (line)
#define MSGSTRFIELD1(line) str##line
static const union msgstr_t
{
  struct
  {
#define _S(n, s) char MSGSTRFIELD(__LINE__)[sizeof (s)];
#include "gai_strerror-strs.h"
#undef _S
  };
  char str[0];
} msgstr =
  {
    {
#define _S(n, s) s,
#include "gai_strerror-strs.h"
#undef _S
    }
  };
static const struct
{
  int16_t code;
  uint16_t idx;
} msgidx[] =
  {
#define _S(n, s) { n, offsetof (union msgstr_t, MSGSTRFIELD (__LINE__)) },
#include "gai_strerror-strs.h"
#undef _S
  };


const char *
gai_strerror (int code)
{
  const char *result = "Unknown error";
  for (size_t i = 0; i < sizeof (msgidx) / sizeof (msgidx[0]); ++i)
    if (msgidx[i].code == code)
      {
	result = msgstr.str + msgidx[i].idx;
	break;
      }

  return _(result);
}
libc_hidden_def (gai_strerror)
