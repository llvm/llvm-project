/* bug 19432: iconv rejects redundant escape sequences in IBM903,
              IBM905, IBM907, and IBM909

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

#include <iconv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

// The longest test input sequence.
#define MAXINBYTES    8
#define MAXOUTBYTES   (MAXINBYTES * MB_LEN_MAX)

/* Verify that a conversion of the INPUT sequence consisting of
   INBYTESLEFT bytes in the encoding specified by the codeset
   named by FROM_SET is successful.
   Return 0 on success, non-zero on iconv() failure.  */

static int
test_ibm93x (const char *from_set, const char *input, size_t inbytesleft)
{
  const char to_set[] = "UTF-8";
  iconv_t cd = iconv_open (to_set, from_set);
  if (cd == (iconv_t) -1)
    {
      printf ("iconv_open(\"%s\", \"%s\"): %s\n",
              from_set, to_set, strerror (errno));
      return 1;
    }

  char output [MAXOUTBYTES];
  size_t outbytesleft = sizeof output;

  char *inbuf = (char*)input;
  char *outbuf = output;

  printf ("iconv(cd, %p, %zu, %p, %zu)\n",
          inbuf, inbytesleft, outbuf, outbytesleft);

  errno = 0;
  size_t ret = iconv (cd, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
  printf ("  ==> %zu: %s\n"
          "  inbuf%+td, inbytesleft=%zu, outbuf%+td, outbytesleft=%zu\n",
          ret, strerror (errno),
          inbuf - input, inbytesleft, outbuf - output, outbytesleft);

  // Return 0 on success, non-zero on iconv() failure.
  return ret == (size_t)-1 || errno;
}

static int
do_test (void)
{
  // State-dependent encodings to exercise.
  static const char* const to_code[] = {
    "IBM930", "IBM933", "IBM935", "IBM937", "IBM939"
  };

  static const size_t ncodesets = sizeof to_code / sizeof *to_code;

  static const struct {
    char txt[MAXINBYTES];
    size_t len;
  } input[] = {
#define DATA(s) { s, sizeof s - 1 }
    /* <SI>: denotes the shift-in 1-byte escape sequence, changing
             the encoder from a sigle-byte encoding to multibyte
       <SO>: denotes the shift-out 1-byte escape sequence, switching
             the encoder from a multibyte to a single-byte state  */

    DATA ("\x0e"),               // <SI> (not redundant)
    DATA ("\x0f"),               // <S0> (redundant with initial state)
    DATA ("\x0e\x0e"),           // <SI><SI>
    DATA ("\x0e\x0f\x0f"),       // <SI><SO><SO>
    DATA ("\x0f\x0f"),           // <SO><SO>
    DATA ("\x0f\x0e\x0e"),       // <SO><SI><SI>
    DATA ("\x0e\x0f\xc7\x0f"),   // <SI><SO><G><SO>
    DATA ("\xc7\x0f")            // <G><SO> (redundant with initial state)
  };

  static const size_t ninputs = sizeof input / sizeof *input;

  int ret = 0;

  size_t i, j;

  /* Iterate over the IBM93x codesets above and exercise each with
     the input sequences above.  */
  for (i = 0; i != ncodesets; ++i)
    for (j = 0; j != ninputs; ++j)
      ret += test_ibm93x (to_code [i], input [i].txt, input [i].len);

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
