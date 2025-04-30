/* Quote a blob so that it can be used in C literals.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <support/support.h>
#include <support/xmemstream.h>

char *
support_quote_blob (const void *blob, size_t length)
{
  struct xmemstream out;
  xopen_memstream (&out);

  const unsigned char *p = blob;
  for (size_t i = 0; i < length; ++i)
    {
      unsigned char ch = p[i];

      /* Use C backslash escapes for those control characters for
         which they are defined.  */
      switch (ch)
        {
          case '\a':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('a', out.out);
            break;
          case '\b':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('b', out.out);
            break;
          case '\f':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('f', out.out);
            break;
          case '\n':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('n', out.out);
            break;
          case '\r':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('r', out.out);
            break;
          case '\t':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('t', out.out);
            break;
          case '\v':
            putc_unlocked ('\\', out.out);
            putc_unlocked ('v', out.out);
            break;
          case '\\':
          case '\'':
          case '\"':
            putc_unlocked ('\\', out.out);
            putc_unlocked (ch, out.out);
            break;
        default:
          if (ch < ' ' || ch > '~')
            /* Use octal sequences because they are fixed width,
               unlike hexadecimal sequences.  */
            fprintf (out.out, "\\%03o", ch);
          else
            putc_unlocked (ch, out.out);
        }
    }

  xfclose_memstream (&out);
  return out.buffer;
}
