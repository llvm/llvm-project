/* Read a line from an nss_files database file.
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

#include <ctype.h>
#include <errno.h>
#include <string.h>

int
__nss_readline (FILE *fp, char *buf, size_t len, off64_t *poffset)
{
  /* We need space for at least one character, the line terminator,
     and the NUL byte.  */
  if (len < 3)
    {
      *poffset = -1;
      __set_errno (ERANGE);
      return ERANGE;
    }

  while (true)
    {
      /* Keep original offset for retries.  */
      *poffset = __ftello64 (fp);

      buf[len - 1] = '\xff';        /* Marker to recognize truncation.  */
      if (__fgets_unlocked (buf, len, fp) == NULL)
        {
          if (__feof_unlocked (fp))
            {
              __set_errno (ENOENT);
              return ENOENT;
            }
          else
            {
              /* Any other error.  Do not return ERANGE in this case
                 because the caller would retry.  */
              if (errno == ERANGE)
                __set_errno (EINVAL);
              return errno;
            }
        }
      else if (buf[len - 1] != '\xff')
        /* The buffer is too small.  Arrange for re-reading the same
           line on the next call.  */
        return __nss_readline_seek (fp, *poffset);

      /* __fgets_unlocked succeeded.  */

      /* Remove leading whitespace.  */
      char *p = buf;
      while (isspace (*p))
        ++p;
      if (*p == '\0' || *p == '#')
        /* Skip empty lines and comments.  */
        continue;
      if (p != buf)
        memmove (buf, p, strlen (p));

      /* Return line to the caller.  */
      return 0;
    }
}
libc_hidden_def (__nss_readline)

int
__nss_readline_seek (FILE *fp, off64_t offset)
{
  if (offset < 0 /* __ftello64 failed.  */
      || __fseeko64 (fp, offset, SEEK_SET) < 0)
    {
      /* Without seeking support, it is not possible to
         re-read the same line, so this is a hard failure.  */
      fseterr_unlocked (fp);
      __set_errno (ESPIPE);
      return ESPIPE;
    }
  else
    {
      __set_errno (ERANGE);
      return ERANGE;
    }
}
