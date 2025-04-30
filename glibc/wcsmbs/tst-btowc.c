/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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

#include <locale.h>
#include <stdio.h>
#include <wchar.h>


/* Currently selected locale.  */
static const char *current_locale;


/* Test which should succeed.  */
static int
ok_test (int c, wint_t expwc)
{
  wint_t wc = btowc (c);

  if (wc != expwc)
    {
      printf ("%s: btowc('%c') failed, returned L'\\x%x' instead of L'\\x%x'\n",
	      current_locale, c, wc, expwc);
      return 1;
    }

  return 0;
}

/* Test which should fail.  */
static int
fail_test (int c)
{
  wint_t wc = btowc (c);

  if (wc != WEOF)
    {
      printf ("%s: btowc('%c') succeded, returned L'\\x%x' instead of WEOF\n",
	      current_locale, c, wc);
      return 1;
    }

  return 0;
}


/* Test EOF handling.  */
static int
eof_test (void)
{
  wint_t wc = btowc (EOF);
  if (wc != WEOF)
    {
      printf ("%s: btowc(EOF) returned L'\\x%x', not WEOF\n",
	      current_locale, wc);
    }

  return 0;
}


/* Test the btowc() function for a few locales with known character sets.  */
static int
do_test (void)
{
  int result = 0;

  current_locale = setlocale (LC_ALL, "en_US.ANSI_X3.4-1968");
  if (current_locale == NULL)
    {
      puts ("cannot set locale \"en_US.ANSI_X3.4-1968\"");
      result = 1;
    }
  else
    {
      int c;

      for (c = 0; c < 128; ++c)
	result |= ok_test (c, c);

      for (c = 128; c < 256; ++c)
	result |= fail_test (c);

      result |= eof_test ();
    }

  current_locale = setlocale (LC_ALL, "de_DE.ISO-8859-1");
  if (current_locale == NULL)
    {
      puts ("cannot set locale \"de_DE.ISO-8859-1\"");
      result = 1;
    }
  else
    {
      int c;

      for (c = 0; c < 256; ++c)
	result |= ok_test (c, c);

      result |= eof_test ();
    }

  current_locale = setlocale (LC_ALL, "de_DE.UTF-8");
  if (current_locale == NULL)
    {
      puts ("cannot set locale \"de_DE.UTF-8\"");
      result = 1;
    }
  else
    {
      int c;

      for (c = 0; c < 128; ++c)
	result |= ok_test (c, c);

      for (c = 128; c < 256; ++c)
	result |= fail_test (c);

      result |= eof_test ();
    }

  current_locale = setlocale (LC_ALL, "hr_HR.ISO-8859-2");
  if (current_locale == NULL)
    {
      puts ("cannot set locale \"hr_HR.ISO-8859-2\"");
      result = 1;
    }
  else
    {
      static const wint_t upper_half[] =
      {
	0x0104, 0x02D8, 0x0141, 0x00A4, 0x013D, 0x015A, 0x00A7, 0x00A8,
	0x0160, 0x015E, 0x0164, 0x0179, 0x00AD, 0x017D, 0x017B, 0x00B0,
	0x0105, 0x02DB, 0x0142, 0x00B4, 0x013E, 0x015B, 0x02C7, 0x00B8,
	0x0161, 0x015F, 0x0165, 0x017A, 0x02DD, 0x017E, 0x017C, 0x0154,
	0x00C1, 0x00C2, 0x0102, 0x00C4, 0x0139, 0x0106, 0x00C7, 0x010C,
	0x00C9, 0x0118, 0x00CB, 0x011A, 0x00CD, 0x00CE, 0x010E, 0x0110,
	0x0143, 0x0147, 0x00D3, 0x00D4, 0x0150, 0x00D6, 0x00D7, 0x0158,
	0x016E, 0x00DA, 0x0170, 0x00DC, 0x00DD, 0x0162, 0x00DF, 0x0155,
	0x00E1, 0x00E2, 0x0103, 0x00E4, 0x013A, 0x0107, 0x00E7, 0x010D,
	0x00E9, 0x0119, 0x00EB, 0x011B, 0x00ED, 0x00EE, 0x010F, 0x0111,
	0x0144, 0x0148, 0x00F3, 0x00F4, 0x0151, 0x00F6, 0x00F7, 0x0159,
	0x016F, 0x00FA, 0x0171, 0x00FC, 0x00FD, 0x0163, 0x02D9
      };
      int c;

      for (c = 0; c < 161; ++c)
	result |= ok_test (c, c);

      for (c = 161; c < 256; ++c)
	result |= ok_test (c, upper_half[c - 161]);

      result |= eof_test ();
    }

  if (result == 0)
    puts ("all OK");

  return result;
}

#include <support/test-driver.c>
