/* Convert a h_errno error code to a string.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/format_nss.h>

#include <support/support.h>

char *
support_format_herrno (int code)
{
  const char *errstr;
  switch (code)
    {
    case HOST_NOT_FOUND:
      errstr = "HOST_NOT_FOUND";
      break;
    case NO_ADDRESS:
      errstr = "NO_ADDRESS";
      break;
    case NO_RECOVERY:
      errstr = "NO_RECOVERY";
      break;
    case TRY_AGAIN:
      errstr = "TRY_AGAIN";
      break;
    default:
      return xasprintf ("<invalid h_errno value %d>\n", code);
    }
  return xstrdup (errstr);
}
