/* Error-checking wrapper for sysconf.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <support/check.h>
#include <support/xunistd.h>

long
xsysconf (int name)
{
  /* Detect errors by a changed errno value, in case -1 is a valid
     value.  Make sure that the caller does not see the zero value for
     errno.  */
  int old_errno = errno;
  errno = 0;
  long result = sysconf (name);
  if (errno != 0)
    FAIL_EXIT1 ("sysconf (%d): %m", name);
  errno = old_errno;
  return result;
}
