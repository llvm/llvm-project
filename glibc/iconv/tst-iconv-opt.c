/* Test iconv's TRANSLIT and IGNORE option handling

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


#include <iconv.h>
#include <locale.h>
#include <errno.h>
#include <string.h>
#include <support/support.h>
#include <support/check.h>


/* Run one iconv test.  Arguments:
   to: destination character set and options
   from: source character set
   input: input string to be converted
   exp_in: expected number of bytes consumed
   exp_ret: expected return value (error or number of irreversible conversions)
   exp_out: expected output string
   exp_err: expected value of `errno' after iconv returns.  */
static void
test_iconv (const char *to, const char *from, char *input, size_t exp_in,
            size_t exp_ret, const char *exp_out, int exp_err)
{
  iconv_t cd;
  char outbuf[500];
  size_t inlen, outlen;
  char *inptr, *outptr;
  size_t n;

  cd = iconv_open (to, from);
  TEST_VERIFY (cd != (iconv_t) -1);

  inlen = strlen (input);
  outlen = sizeof (outbuf);
  inptr = input;
  outptr = outbuf;

  errno = 0;
  n = iconv (cd, &inptr, &inlen, &outptr, &outlen);

  TEST_COMPARE (n, exp_ret);
  TEST_VERIFY (inptr == input + exp_in);
  TEST_COMPARE (errno, exp_err);
  TEST_COMPARE_BLOB (outbuf, outptr - outbuf, exp_out, strlen (exp_out));
  TEST_VERIFY (iconv_close (cd) == 0);
}


/* We test option parsing by converting UTF-8 inputs to ASCII under various
   option combinations. The UTF-8 inputs fall into three categories:
   - ASCII-only,
   - non-ASCII,
   - non-ASCII with invalid UTF-8 characters.  */

/* 1.  */
char ascii[] = "Just some ASCII text";

/* 2. Valid UTF-8 input and some corresponding expected outputs with various
   options.  The two non-ASCII characters below are accented alphabets:
   an `a' then an `o'.  */
char utf8[] = "UTF-8 text with \u00E1 couple \u00F3f non-ASCII characters";
char u2a[] = "UTF-8 text with ";
char u2a_translit[] = "UTF-8 text with a couple of non-ASCII characters";
char u2a_ignore[] = "UTF-8 text with  couple f non-ASCII characters";

/* 3. Invalid UTF-8 input and some corresponding expected outputs.  \xff is
   invalid UTF-8. It's followed by some valid but non-ASCII UTF-8.  */
char iutf8[] = "Invalid UTF-8 \xff\u27E6text\u27E7";
char iu2a[] = "Invalid UTF-8 ";
char iu2a_ignore[] = "Invalid UTF-8 text";
char iu2a_both[] = "Invalid UTF-8 [|text|]";

/* 4. Another invalid UTF-8 input and corresponding expected outputs. This time
   the valid non-ASCII UTF-8 characters appear before the invalid \xff.  */
char jutf8[] = "Invalid \u27E6UTF-8\u27E7 \xfftext";
char ju2a[] = "Invalid ";
char ju2a_translit[] = "Invalid [|UTF-8|] ";
char ju2a_ignore[] = "Invalid UTF-8 text";
char ju2a_both[] = "Invalid [|UTF-8|] text";

/* We also test option handling for character set names that have the form
   "A/B".  In this test, we test conversions "ISO-10646/UTF-8", and either
   ISO-8859-1 or ASCII.  */

/* 5. Accented 'A' and 'a' characters in ISO-8859-1 and UTF-8, and an
   equivalent ASCII transliteration.  */
char iso8859_1_a[] = {0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, /* Accented A's.  */
                      0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, /* Accented a's.  */
                      0x00};
char utf8_a[] = "\u00C0\u00C1\u00C2\u00C3\u00C4\u00C5"
                "\u00E0\u00E1\u00E2\u00E3\u00E4\u00E5";
char ascii_a[] = "AAAAAAaaaaaa";

/* 6. An invalid ASCII string where [0] is invalid and [1] is '~'.  */
char iascii [] = {0x80, '~', '\0'};
char empty[] = "";
char ia2u_ignore[] = "~";

static int
do_test (void)
{
  xsetlocale (LC_ALL, "en_US.UTF-8");


  /* 0. iconv_open should gracefully fail for invalid character sets.  */

  TEST_VERIFY (iconv_open ("INVALID", "UTF-8") == (iconv_t) -1);
  TEST_VERIFY (iconv_open ("UTF-8", "INVALID") == (iconv_t) -1);
  TEST_VERIFY (iconv_open ("INVALID", "INVALID") == (iconv_t) -1);


  /* 1. ASCII-only UTF-8 input should convert to ASCII with no changes:  */

  test_iconv ("ASCII", "UTF-8", ascii, strlen (ascii), 0, ascii, 0);
  test_iconv ("ASCII//", "UTF-8", ascii, strlen (ascii), 0, ascii, 0);
  test_iconv ("ASCII//TRANSLIT", "UTF-8", ascii, strlen (ascii), 0, ascii, 0);
  test_iconv ("ASCII//TRANSLIT//", "UTF-8", ascii, strlen (ascii), 0, ascii,
              0);
  test_iconv ("ASCII//IGNORE", "UTF-8", ascii, strlen (ascii), 0, ascii, 0);
  test_iconv ("ASCII//IGNORE//", "UTF-8", ascii, strlen (ascii), 0, ascii, 0);


  /* 2. Valid UTF-8 input with non-ASCII characters:  */

  /* EILSEQ when converted to ASCII.  */
  test_iconv ("ASCII", "UTF-8", utf8, strlen (u2a), (size_t) -1, u2a, EILSEQ);

  /* Converted without error with TRANSLIT enabled.  */
  test_iconv ("ASCII//TRANSLIT", "UTF-8", utf8, strlen (utf8), 2, u2a_translit,
              0);

  /* EILSEQ with IGNORE enabled.  Non-ASCII chars dropped from output.  */
  test_iconv ("ASCII//IGNORE", "UTF-8", utf8, strlen (utf8), (size_t) -1,
              u2a_ignore, EILSEQ);

  /* With TRANSLIT and IGNORE enabled, transliterated without error.  We test
     four combinations.  */

  test_iconv ("ASCII//TRANSLIT,IGNORE", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);
  test_iconv ("ASCII//TRANSLIT//IGNORE", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);
  /* Due to bug 19519, iconv was ignoring TRANSLIT for the following input.  */
  test_iconv ("ASCII//IGNORE//TRANSLIT", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);

  /* Misspellings of TRANSLIT and IGNORE are ignored, but conversion still
     works while respecting any other correctly spelled options.  */

  test_iconv ("ASCII//T", "UTF-8", utf8, strlen (u2a), (size_t) -1, u2a,
              EILSEQ);
  test_iconv ("ASCII//TRANSLITERATE", "UTF-8", utf8, strlen (u2a), (size_t) -1,
              u2a, EILSEQ);
  test_iconv ("ASCII//I", "UTF-8", utf8, strlen (u2a), (size_t) -1, u2a,
              EILSEQ);
  test_iconv ("ASCII//IGNORED", "UTF-8", utf8, strlen (u2a), (size_t) -1, u2a,
              EILSEQ);
  test_iconv ("ASCII//TRANSLITERATE//IGNORED", "UTF-8", utf8, strlen (u2a),
              (size_t) -1, u2a, EILSEQ);
  test_iconv ("ASCII//IGNORED,TRANSLITERATE", "UTF-8", utf8, strlen (u2a),
              (size_t) -1, u2a, EILSEQ);
  test_iconv ("ASCII//T//I", "UTF-8", utf8, strlen (u2a), (size_t) -1, u2a,
              EILSEQ);

  test_iconv ("ASCII//TRANSLIT//I", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);
  /* Due to bug 19519, iconv was ignoring TRANSLIT for the following input.  */
  test_iconv ("ASCII//I//TRANSLIT", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);
  test_iconv ("ASCII//IGNORED,TRANSLIT", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);
  test_iconv ("ASCII//TRANSLIT,IGNORED", "UTF-8", utf8, strlen (utf8), 2,
              u2a_translit, 0);

  test_iconv ("ASCII//IGNORE,T", "UTF-8", utf8, strlen (utf8), (size_t) -1,
              u2a_ignore, EILSEQ);
  test_iconv ("ASCII//T,IGNORE", "UTF-8", utf8, strlen (utf8), (size_t) -1,
              u2a_ignore, EILSEQ);
  /* Due to bug 19519, iconv was ignoring IGNORE for the following input.  */
  test_iconv ("ASCII//TRANSLITERATE//IGNORE", "UTF-8", utf8, strlen (utf8),
              (size_t) -1, u2a_ignore, EILSEQ);
  test_iconv ("ASCII//IGNORE//TRANSLITERATE", "UTF-8", utf8, strlen (utf8),
              (size_t) -1, u2a_ignore, EILSEQ);


  /* 3. Invalid UTF-8 followed by some valid non-ASCII UTF-8 characters:  */

  /* EILSEQ; output is truncated at the first invalid UTF-8 character.  */
  test_iconv ("ASCII", "UTF-8", iutf8, strlen (iu2a), (size_t) -1, iu2a,
              EILSEQ);

  /* With TRANSLIT enabled: EILSEQ; output still truncated at the first invalid
     UTF-8 character.  */
  test_iconv ("ASCII//TRANSLIT", "UTF-8", iutf8, strlen (iu2a), (size_t) -1,
              iu2a, EILSEQ);

  /* With IGNORE enabled: EILSEQ; output omits invalid UTF-8 characters and
     valid UTF-8 non-ASCII characters.  */
  test_iconv ("ASCII//IGNORE", "UTF-8", iutf8, strlen (iutf8), (size_t) -1,
              iu2a_ignore, EILSEQ);

  /* With TRANSLIT and IGNORE enabled, output omits only invalid UTF-8
     characters and transliterates valid non-ASCII UTF-8 characters.  We test
     four combinations.  */

  test_iconv ("ASCII//TRANSLIT,IGNORE", "UTF-8", iutf8, strlen (iutf8), 2,
              iu2a_both, 0);
  /* Due to bug 19519, iconv was ignoring IGNORE for the following input.  */
  test_iconv ("ASCII//TRANSLIT//IGNORE", "UTF-8", iutf8, strlen (iutf8), 2,
              iu2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT", "UTF-8", iutf8, strlen (iutf8), 2,
              iu2a_both, 0);
  /* Due to bug 19519, iconv was ignoring TRANSLIT for the following input.  */
  test_iconv ("ASCII//IGNORE//TRANSLIT", "UTF-8", iutf8, strlen (iutf8), 2,
              iu2a_both, 0);


  /* 4. Invalid UTF-8 with valid non-ASCII UTF-8 chars appearing first:  */

  /* EILSEQ; output is truncated at the first non-ASCII character.  */
  test_iconv ("ASCII", "UTF-8", jutf8, strlen (ju2a), (size_t) -1, ju2a,
              EILSEQ);

  /* With TRANSLIT enabled: EILSEQ; output now truncated at the first invalid
     UTF-8 character.  */
  test_iconv ("ASCII//TRANSLIT", "UTF-8", jutf8, strlen (jutf8) - 5,
              (size_t) -1, ju2a_translit, EILSEQ);
  test_iconv ("ASCII//translit", "UTF-8", jutf8, strlen (jutf8) - 5,
              (size_t) -1, ju2a_translit, EILSEQ);

  /* With IGNORE enabled: EILSEQ; output omits invalid UTF-8 characters and
     valid UTF-8 non-ASCII characters.  */
  test_iconv ("ASCII//IGNORE", "UTF-8", jutf8, strlen (jutf8), (size_t) -1,
              ju2a_ignore, EILSEQ);
  test_iconv ("ASCII//ignore", "UTF-8", jutf8, strlen (jutf8), (size_t) -1,
              ju2a_ignore, EILSEQ);

  /* With TRANSLIT and IGNORE enabled, output omits only invalid UTF-8
     characters and transliterates valid non-ASCII UTF-8 characters.  We test
     several combinations.  */

  test_iconv ("ASCII//TRANSLIT,IGNORE", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  /* Due to bug 19519, iconv was ignoring IGNORE for the following input.  */
  test_iconv ("ASCII//TRANSLIT//IGNORE", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  /* Due to bug 19519, iconv was ignoring TRANSLIT for the following input.  */
  test_iconv ("ASCII//IGNORE//TRANSLIT", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//translit,ignore", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  /* Trailing whitespace and separators should be ignored.  */
  test_iconv ("ASCII//IGNORE,TRANSLIT ", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT/", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT//", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT,", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT,,", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);
  test_iconv ("ASCII//IGNORE,TRANSLIT /,", "UTF-8", jutf8, strlen (jutf8), 2,
              ju2a_both, 0);

  /* TRANSLIT or IGNORE suffixes in fromcode should be ignored.  */
  test_iconv ("ASCII", "UTF-8//TRANSLIT", jutf8, strlen (ju2a), (size_t) -1,
              ju2a, EILSEQ);
  test_iconv ("ASCII", "UTF-8//IGNORE", jutf8, strlen (ju2a), (size_t) -1,
              ju2a, EILSEQ);
  test_iconv ("ASCII", "UTF-8//TRANSLIT,IGNORE", jutf8, strlen (ju2a),
              (size_t) -1, ju2a, EILSEQ);


  /* 5. Charset names of the form "A/B/":  */

  /* ISO-8859-1 is converted to UTF-8 without needing transliteration.  */
  test_iconv ("ISO-10646/UTF-8", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8/", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8/IGNORE", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8//IGNORE", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8/TRANSLIT", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8//TRANSLIT", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8//TRANSLIT/IGNORE", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8//TRANSLIT//IGNORE", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);
  test_iconv ("ISO-10646/UTF-8/TRANSLIT,IGNORE", "ISO-8859-1", iso8859_1_a,
              strlen (iso8859_1_a), 0, utf8_a, 0);

  /* UTF-8 with accented A's is converted to ASCII with transliteration.  */
  test_iconv ("ASCII", "ISO-10646/UTF-8", utf8_a,
              0, (size_t) -1, empty, EILSEQ);
  test_iconv ("ASCII//IGNORE", "ISO-10646/UTF-8", utf8_a,
              strlen (utf8_a), (size_t) -1, empty, EILSEQ);
  test_iconv ("ASCII//TRANSLIT", "ISO-10646/UTF-8", utf8_a,
              strlen (utf8_a), 12, ascii_a, 0);

  /* Invalid ASCII is converted to UTF-8 only with IGNORE.  */
  test_iconv ("ISO-10646/UTF-8", "ASCII", iascii, strlen (empty), (size_t) -1,
              empty, EILSEQ);
  test_iconv ("ISO-10646/UTF-8/TRANSLIT", "ASCII", iascii, strlen (empty),
              (size_t) -1, empty, EILSEQ);
  test_iconv ("ISO-10646/UTF-8/IGNORE", "ASCII", iascii, strlen (iascii),
              (size_t) -1, ia2u_ignore, EILSEQ);
  test_iconv ("ISO-10646/UTF-8/TRANSLIT,IGNORE", "ASCII", iascii,
              strlen (iascii), (size_t) -1, ia2u_ignore, EILSEQ);
  /* Due to bug 19519, iconv was ignoring IGNORE for the following three
     inputs: */
  test_iconv ("ISO-10646/UTF-8/TRANSLIT/IGNORE", "ASCII", iascii,
              strlen (iascii), (size_t) -1, ia2u_ignore, EILSEQ);
  test_iconv ("ISO-10646/UTF-8//TRANSLIT,IGNORE", "ASCII", iascii,
              strlen (iascii), (size_t) -1, ia2u_ignore, EILSEQ);
  test_iconv ("ISO-10646/UTF-8//TRANSLIT//IGNORE", "ASCII", iascii,
              strlen (iascii), (size_t) -1, ia2u_ignore, EILSEQ);

  return 0;
}

#include <support/test-driver.c>
