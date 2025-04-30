/* Test for open_memstream implementation.
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

#include <wchar.h>

/* Straighforward implementation so tst-memstream3 could use check
   fwrite on open_memstream.  */
static size_t
fwwrite (const void *ptr, size_t size, size_t nmemb, FILE *arq)
{
  const wchar_t *wcs = (const wchar_t*) (ptr);
  for (size_t s = 0; s < size; s++)
    {
      for (size_t n = 0; n < nmemb; n++)
        if (fputwc (wcs[n], arq) == WEOF)
          return n;
    }
  return size * nmemb;
}

#define CHAR_T wchar_t
#define W(o) L##o
#define OPEN_MEMSTREAM open_wmemstream
#define PRINTF wprintf
#define FWRITE_FUNC fwwrite
#define FPUTC  fputwc
#define STRCMP wcscmp

#include "tst-memstream3.c"
