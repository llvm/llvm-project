/* Classify a domain name for IDNA purposes.
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

#include <errno.h>
#include <inet/net-internal.h>
#include <stdbool.h>
#include <string.h>
#include <wchar.h>

enum idna_name_classification
__idna_name_classify (const char *name)
{
  mbstate_t mbs;
  memset (&mbs, 0, sizeof (mbs));
  const char *p = name;
  const char *end = p + strlen (p) + 1;
  bool nonascii = false;
  bool backslash = false;
  while (true)
    {
      wchar_t wc;
      size_t result = mbrtowc (&wc, p, end - p, &mbs);
      if (result == 0)
        /* NUL terminator was reached.  */
        break;
      else if (result == (size_t) -2)
        /* Incomplete trailing multi-byte character.  This is an
           encoding error becaue we received the full name.  */
        return idna_name_encoding_error;
      else if (result == (size_t) -1)
        {
          /* Other error, including EILSEQ.  */
          if (errno == EILSEQ)
            return idna_name_encoding_error;
          else if (errno == ENOMEM)
            return idna_name_memory_error;
          else
            return idna_name_error;
        }
      else
        {
          /* A wide character was decoded.  */
          p += result;
          if (wc == L'\\')
            backslash = true;
          else if (wc > 127)
            nonascii = true;
        }
    }

  if (nonascii)
    {
      if (backslash)
        return idna_name_nonascii_backslash;
      else
        return idna_name_nonascii;
    }
  else
    return idna_name_ascii;
}
