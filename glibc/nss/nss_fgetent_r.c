/* Generic implementation of fget*ent_r.
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

#include <errno.h>
#include <nss_files.h>
#include <stdbool.h>

int
__nss_fgetent_r (FILE *fp, void *result, char *buffer, size_t buffer_length,
                 nss_files_parse_line parser)
{
  int ret;

  _IO_flockfile (fp);

  while (true)
    {
      off64_t original_offset;
      ret = __nss_readline (fp, buffer, buffer_length, &original_offset);
      if (ret == 0)
        {
          /* Parse the line into *RESULT.  */
          ret = parser (buffer, result,
                        (struct parser_data *) buffer, buffer_length, &errno);

          /* Translate the result code from the parser into an errno
             value.  Also seeks back to the start of the line if
             necessary.  */
          ret = __nss_parse_line_result (fp, original_offset, ret);

          if (ret == EINVAL)
            /* Skip over malformed lines.  */
            continue;
        }
      break;
    }

  _IO_funlockfile (fp);

  return ret;
}
