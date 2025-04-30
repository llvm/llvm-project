/* Implementation of __nss_parse_line_result.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <nss_files.h>

#include <assert.h>
#include <errno.h>

int
__nss_parse_line_result (FILE *fp, off64_t offset, int parse_line_result)
{
  assert (parse_line_result >= -1 && parse_line_result <= 1);

  switch (__builtin_expect (parse_line_result, 1))
    {
    case 1:
      /* Sucess.  */
      return 0;
    case 0:
      /* Parse error.  */
      __set_errno (EINVAL);
      return EINVAL;
    case -1:
      /* Out of buffer space.  */
      return __nss_readline_seek (fp, offset);

      default:
        __builtin_unreachable ();
    }
}
libc_hidden_def (__nss_parse_line_result)
