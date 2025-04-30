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

#include <limits.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#define ISLOWER(c) ('a' <= (c) && (c) <= 'z')
#define TOUPPER(c) (ISLOWER(c) ? 'A' + ((c) - 'a') : (c))
#define XOR(e,f) (((e) && !(f)) || (!(e) && (f)))

#ifdef	__GNUC__
__inline
#endif
static void
print_char (unsigned char c)
{
  printf("%d/", (int) c);
  if (isgraph(c))
    printf("'%c'", c);
  else
    printf("'\\%.3o'", c);
}

int
main (int argc, char **argv)
{
  unsigned short int c;
  int lose = 0;

#define TRYEM do {							      \
      TRY (isascii);							      \
      TRY (isalnum);							      \
      TRY (isalpha);							      \
      TRY (iscntrl);							      \
      TRY (isdigit);							      \
      TRY (isgraph);							      \
      TRY (islower);							      \
      TRY (isprint);							      \
      TRY (ispunct);							      \
      TRY (isspace);							      \
      TRY (isupper);							      \
      TRY (isxdigit);							      \
      TRY (isblank);							      \
    } while (0)

  for (c = 0; c <= UCHAR_MAX; ++c)
    {
      print_char (c);

      if (XOR (islower (c), ISLOWER (c)) || toupper (c) != TOUPPER (c))
	{
	  fputs (" BOGUS", stdout);
	  ++lose;
	}

#define TRY(isfoo) if (isfoo (c)) fputs (" " #isfoo, stdout)
      TRYEM;
#undef TRY

      fputs("; lower = ", stdout);
      print_char(tolower(c));
      fputs("; upper = ", stdout);
      print_char(toupper(c));
      putchar('\n');
    }

  fputs ("EOF", stdout);
  if (tolower (EOF) != EOF)
    {
      ++lose;
      printf (" tolower BOGUS %d;", tolower (EOF));
    }
  if (toupper (EOF) != EOF)
    {
      ++lose;
      printf (" toupper BOGUS %d;", toupper (EOF));
    }

#define TRY(isfoo) if (isfoo (EOF)) fputs (" " #isfoo, stdout), ++lose
  TRYEM;
#undef TRY

  return lose ? EXIT_FAILURE : EXIT_SUCCESS;
}
