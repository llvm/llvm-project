/* Test some mathematical operator transliterations (BZ #23132)

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <iconv.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>

static int
do_test (void)
{
  iconv_t cd;

  /* str[] = "⟦ ⟧ ⟨ ⟩"
             " ⟬ ⟭ ⦀"
             " ⦃ ⦄ ⦅ ⦆"
             " ⦇ ⦈ ⦉ ⦊"
             " ⧣ ⧥ ⧵ ⧸ ⧹"
             " ⧼ ⧽ ⧾ ⧿";  */

  const char str[] = "\u27E6 \u27E7 \u27E8 \u27E9"
                     " \u27EC \u27ED \u2980"
                     " \u2983 \u2984 \u2985 \u2986"
                     " \u2987 \u2988 \u2989 \u298A"
                     " \u29E3 \u29E5 \u29F5 \u29F8 \u29F9"
                     " \u29FC \u29FD \u29FE \u29FF";

  const char expected[] = "[| |] < >"
                          " (( )) |||"
                          " {| |} (( ))"
                          " (| |) <| |>"
                          " # # \\ / \\"
                          " < > + -";

  char *inptr = (char *) str;
  size_t inlen = strlen (str) + 1;
  char outbuf[500];
  char *outptr = outbuf;
  size_t outlen = sizeof (outbuf);
  int result = 0;
  size_t n;

  if (setlocale (LC_ALL, "en_US.UTF-8") == NULL)
    FAIL_EXIT1 ("setlocale failed");

  cd = iconv_open ("ASCII//TRANSLIT", "UTF-8");
  if (cd == (iconv_t) -1)
    FAIL_EXIT1 ("iconv_open failed");

  n = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  if (n != 24)
    {
      if (n == (size_t) -1)
        printf ("iconv() returned error: %m\n");
      else
        printf ("iconv() returned %Zd, expected 24\n", n);
      result = 1;
    }
  if (inlen != 0)
    {
      puts ("not all input consumed");
      result = 1;
    }
  else if (inptr - str != strlen (str) + 1)
    {
      printf ("inptr wrong, advanced by %td\n", inptr - str);
      result = 1;
    }
  if (memcmp (outbuf, expected, sizeof (expected)) != 0)
    {
      printf ("result wrong: \"%.*s\", expected: \"%s\"\n",
              (int) (sizeof (outbuf) - outlen), outbuf, expected);
      result = 1;
    }
  else if (outlen != sizeof (outbuf) - sizeof (expected))
    {
      printf ("outlen wrong: %Zd, expected %Zd\n", outlen,
              sizeof (outbuf) - 15);
      result = 1;
    }
  else
    printf ("output is \"%s\" which is OK\n", outbuf);

  return result;
}

#include <support/test-driver.c>
