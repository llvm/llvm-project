/* Test for wchar_t/multi-byte conversion and precision in vfprintf.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/test-driver.h>
#include <wchar.h>

#define DYNARRAY_STRUCT str
#define DYNARRAY_ELEMENT char
#define DYNARRAY_PREFIX str_
#include <malloc/dynarray-skeleton.c>

#define DYNARRAY_STRUCT wstr
#define DYNARRAY_ELEMENT wchar_t
#define DYNARRAY_PREFIX wstr_
#include <malloc/dynarray-skeleton.c>

#define DYNARRAY_STRUCT len
#define DYNARRAY_ELEMENT size_t
#define DYNARRAY_PREFIX len_
#include <malloc/dynarray-skeleton.c>

/* This should be larger than the internal buffer in vfprintf.  The
   constant needs to be kept in sync with the format strings in
   test_mbs_long and test_wide_long.  */
enum { WIDE_STRING_LENGTH = 1000 };

/* Creates two large, random strings used for truncation testing.
   After the call, *MBS will be encoded in UTF-8, and *WIDE will
   contain the same string in the internal UCS-32 encoding.  Both
   strings are null-terminated.  The array *LENGTH counts the number
   of multi-byte characters for each prefix string of *WIDE: The first
   N wide characters of *WIDE correspond the first (*LENGTH)[N] bytes
   of *MBS.  The caller should deallocate all three arrays using
   free.  */
static void
make_random_string (char **mbs, wchar_t **wide, size_t **length)
{
  struct str str;
  str_init (&str);
  struct wstr wstr;
  wstr_init (&wstr);
  struct len len;
  len_init (&len);

  for (int i = 0; i < WIDE_STRING_LENGTH; ++i)
    {
      len_add (&len, str_size (&str));
      /* Cover some multi-byte UTF-8 sequences.  Avoid the null
         character.  */
      uint32_t ch = 1 + (rand () % 521);
      wstr_add (&wstr, ch);

      /* Limited UTF-8 conversion.  */
      if (ch <= 127)
        str_add (&str, ch);
      else
        {
          /* We only implement two-byte sequences.  */
          uint32_t first = ch >> 6;
          TEST_VERIFY (first < 32);
          str_add (&str, 0xC0 | first);
          str_add (&str, 0x80 | (ch & 0x3f));
        }
    }
  len_add (&len, str_size (&str));
  wstr_add (&wstr, L'\0');
  str_add (&str, '\0');

  *mbs = str_finalize (&str, NULL);
  TEST_VERIFY_EXIT (*mbs != NULL);
  *wide = wstr_finalize (&wstr, NULL);
  TEST_VERIFY_EXIT (*wide != NULL);
  *length = len_finalize (&len, NULL);
  TEST_VERIFY_EXIT (*length != NULL);
}

/* snprintf tests (multi-byte result).  */
static void
test_mbs_result (void)
{
  char buf[200];

  /* ASCII wide string.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%ls", L"xyz") == 3);
  TEST_VERIFY (strcmp (buf, "xyz") == 0);

  /* Unicode wide string.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%.1ls", L"x\u00DFz") == 1);
  TEST_VERIFY (strcmp (buf, "x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%.2ls", L"x\u00DFz") == 1);
  TEST_VERIFY (strcmp (buf, "x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%.3ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9F") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions with width 2, right-justified.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%2.1ls", L"x\u00DFz") == 2);
  TEST_VERIFY (strcmp (buf, " x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%2.2ls", L"x\u00DFz") == 2);
  TEST_VERIFY (strcmp (buf, " x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%2.3ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9F") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%2.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%2.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions with width 2, left-justified.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-2.1ls", L"x\u00DFz") == 2);
  TEST_VERIFY (strcmp (buf, "x ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-2.2ls", L"x\u00DFz") == 2);
  TEST_VERIFY (strcmp (buf, "x ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-2.3ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9F") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-2.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-2.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions with width 3, right-justified.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%3.1ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "  x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%3.2ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "  x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%3.3ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9F") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%3.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%3.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions with width 3, left-justified.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-3.1ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x  ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-3.2ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x  ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-3.3ls", L"x\u00DFz") == 3);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9F") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-3.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-3.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions with width 4, right-justified.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%4.1ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "   x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%4.2ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "   x") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%4.3ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, " x\xC3\x9F") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%4.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%4.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);

  /* Varying precisions with width 4, left-justified.  */
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-4.1ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x   ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-4.2ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x   ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-4.3ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9F ") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-4.4ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
  memset (buf, '@', sizeof (buf));
  TEST_VERIFY (snprintf (buf, sizeof (buf), "%-4.5ls", L"x\u00DFz") == 4);
  TEST_VERIFY (strcmp (buf, "x\xC3\x9Fz") == 0);
}

/* swprintf tests (wide string result).  */
static void
test_wide_result (void)
{
  enum { size = 20 };
  wchar_t buf[20];

  /* ASCII wide string.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%s", "xyz") == 3);
  TEST_VERIFY (wcscmp (buf, L"xyz") == 0);

  /* Unicode wide string.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);

  /* Varying precisions.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%.1s", "x\xC3\x9Fz") == 1);
  TEST_VERIFY (wcscmp (buf, L"x") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%.2s", "x\xC3\x9Fz") == 2);
  TEST_VERIFY (wcscmp (buf, L"x\u00DF") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%.3s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%.4s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);

  /* Varying precisions with width 2, right-justified.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%2.1s", "x\xC3\x9Fz") == 2);
  TEST_VERIFY (wcscmp (buf, L" x") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%2.2s", "x\xC3\x9Fz") == 2);
  TEST_VERIFY (wcscmp (buf, L"x\u00DF") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%2.3s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%2.4s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);

  /* Varying precisions with width 2, left-justified.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-2.1s", "x\xC3\x9Fz") == 2);
  TEST_VERIFY (wcscmp (buf, L"x ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-2.2s", "x\xC3\x9Fz") == 2);
  TEST_VERIFY (wcscmp (buf, L"x\u00DF") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-2.3s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-2.4s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);

  /* Varying precisions with width 3, right-justified.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%3.1s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"  x") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%3.2s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L" x\u00DF") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%3.3s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%3.4s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);

  /* Varying precisions with width 3, left-justified.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-3.1s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x  ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-3.2s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DF ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-3.3s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-3.4s", "x\xC3\x9Fz") == 3);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz") == 0);

  /* Varying precisions with width 4, right-justified.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%4.1s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"   x") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%4.2s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"  x\u00DF") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%4.3s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L" x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%4.4s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L" x\u00DFz") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%4.5s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L" x\u00DFz") == 0);

  /* Varying precisions with width 4, left-justified.  */
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-4.1s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"x   ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-4.2s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"x\u00DF  ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-4.3s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-4.4s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz ") == 0);
  wmemset (buf, '@', size);
  TEST_VERIFY (swprintf (buf, size, L"%-4.5s", "x\xC3\x9Fz") == 4);
  TEST_VERIFY (wcscmp (buf, L"x\u00DFz ") == 0);
}

/* Test with long strings and multi-byte result.  */
static void
test_mbs_long (const char *mbs, const wchar_t *wide, const size_t *length)
{
  char buf[4000];
  _Static_assert (sizeof (buf) > 3 * WIDE_STRING_LENGTH,
                  "buffer size consistent with string length");
  const char *suffix = "||TERM";
  TEST_VERIFY_EXIT (sizeof (buf)
                    > length[WIDE_STRING_LENGTH] + strlen (suffix));

  /* Test formatting of the entire string.  */
  {
    int ret = snprintf (buf, sizeof (buf), "%ls%s", wide, suffix);
    TEST_VERIFY (ret == length[WIDE_STRING_LENGTH] + strlen (suffix));
    TEST_VERIFY (memcmp (buf, mbs, length[WIDE_STRING_LENGTH]) == 0);
    TEST_VERIFY (strcmp (buf + length[WIDE_STRING_LENGTH], suffix) == 0);

    /* Left-justified string, printed in full.  */
    ret = snprintf (buf, sizeof (buf), "%-3500ls%s", wide, suffix);
    TEST_VERIFY (ret == 3500 + strlen (suffix));
    TEST_VERIFY (memcmp (buf, mbs, length[WIDE_STRING_LENGTH]) == 0);
    for (size_t i = length[WIDE_STRING_LENGTH]; i < 3500; ++i)
      TEST_VERIFY (buf[i] == ' ');
    TEST_VERIFY (strcmp (buf + 3500, suffix) == 0);

    /* Right-justified string, printed in full.   */
    ret = snprintf (buf, sizeof (buf), "%3500ls%s", wide, suffix);
    TEST_VERIFY (ret == 3500 + strlen (suffix));
    size_t padding = 3500 - length[WIDE_STRING_LENGTH];
    for (size_t i = 0; i < padding; ++i)
      TEST_VERIFY (buf[i] == ' ');
    TEST_VERIFY (memcmp (buf + padding, mbs, length[WIDE_STRING_LENGTH]) == 0);
    TEST_VERIFY (strcmp (buf + 3500, suffix) == 0);
  }

  size_t wide_characters_converted = 0;
  for (int mbs_len = 0; mbs_len <= length[WIDE_STRING_LENGTH] + 1;
       ++mbs_len)
    {
      if (wide_characters_converted < WIDE_STRING_LENGTH
          && mbs_len >= length[wide_characters_converted + 1])
        /* The requested prefix length contains room for another wide
           character.  */
        ++wide_characters_converted;
      if (test_verbose > 0)
        printf ("info: %s: mbs_len=%d wide_chars_converted=%zu length=%zu\n",
                __func__, mbs_len, wide_characters_converted,
                length[wide_characters_converted]);
      TEST_VERIFY (length[wide_characters_converted] <= mbs_len);
      TEST_VERIFY (wide_characters_converted == 0
                   || length[wide_characters_converted - 1] < mbs_len);

      int ret = snprintf (buf, sizeof (buf), "%.*ls%s", mbs_len, wide, suffix);
      TEST_VERIFY (ret == length[wide_characters_converted] + strlen (suffix));
      TEST_VERIFY (memcmp (buf, mbs, length[wide_characters_converted]) == 0);
      TEST_VERIFY (strcmp (buf + length[wide_characters_converted],
                           suffix) == 0);

      /* Left-justified string, printed in full.  */
      if (test_verbose)
        printf ("info: %s: left-justified\n", __func__);
      ret = snprintf (buf, sizeof (buf), "%-3500.*ls%s",
                      mbs_len, wide, suffix);
      TEST_VERIFY (ret == 3500 + strlen (suffix));
      TEST_VERIFY (memcmp (buf, mbs, length[wide_characters_converted]) == 0);
      for (size_t i = length[wide_characters_converted]; i < 3500; ++i)
        TEST_VERIFY (buf[i] == ' ');
      TEST_VERIFY (strcmp (buf + 3500, suffix) == 0);

      /* Right-justified string, printed in full.   */
      if (test_verbose)
        printf ("info: %s: right-justified\n", __func__);
      ret = snprintf (buf, sizeof (buf), "%3500.*ls%s", mbs_len, wide, suffix);
      TEST_VERIFY (ret == 3500 + strlen (suffix));
      size_t padding = 3500 - length[wide_characters_converted];
      for (size_t i = 0; i < padding; ++i)
        TEST_VERIFY (buf[i] == ' ');
      TEST_VERIFY (memcmp (buf + padding, mbs,
                           length[wide_characters_converted]) == 0);
      TEST_VERIFY (strcmp (buf + 3500, suffix) == 0);
    }
}

/* Test with long strings and wide string result.  */
static void
test_wide_long (const char *mbs, const wchar_t *wide, const size_t *length)
{
  wchar_t buf[2000];
  _Static_assert (sizeof (buf) > sizeof (wchar_t) * WIDE_STRING_LENGTH,
                  "buffer size consistent with string length");
  const wchar_t *suffix = L"||TERM";
  TEST_VERIFY_EXIT (sizeof (buf)
                    > length[WIDE_STRING_LENGTH] + wcslen (suffix));

  /* Test formatting of the entire string.  */
  {
    int ret = swprintf (buf, sizeof (buf), L"%s%ls", mbs, suffix);
    TEST_VERIFY (ret == WIDE_STRING_LENGTH + wcslen (suffix));
    TEST_VERIFY (wmemcmp (buf, wide, WIDE_STRING_LENGTH) == 0);
    TEST_VERIFY (wcscmp (buf + WIDE_STRING_LENGTH, suffix) == 0);

    /* Left-justified string, printed in full.  */
    ret = swprintf (buf, sizeof (buf), L"%-1500s%ls", mbs, suffix);
    TEST_VERIFY (ret == 1500 + wcslen (suffix));
    TEST_VERIFY (wmemcmp (buf, wide, WIDE_STRING_LENGTH) == 0);
    for (size_t i = WIDE_STRING_LENGTH; i < 1500; ++i)
      TEST_VERIFY (buf[i] == L' ');
    TEST_VERIFY (wcscmp (buf + 1500, suffix) == 0);

    /* Right-justified string, printed in full.   */
    ret = swprintf (buf, sizeof (buf), L"%1500s%ls", mbs, suffix);
    TEST_VERIFY (ret == 1500 + wcslen (suffix));
    size_t padding = 1500 - WIDE_STRING_LENGTH;
    for (size_t i = 0; i < padding; ++i)
      TEST_VERIFY (buf[i] == ' ');
    TEST_VERIFY (wmemcmp (buf + padding, wide, WIDE_STRING_LENGTH) == 0);
    TEST_VERIFY (wcscmp (buf + 1500, suffix) == 0);
  }

  for (int wide_len = 0; wide_len <= WIDE_STRING_LENGTH + 1; ++wide_len)
    {
      size_t actual_wide_len;
      if (wide_len < WIDE_STRING_LENGTH)
        actual_wide_len = wide_len;
      else
        actual_wide_len = WIDE_STRING_LENGTH;
      if (test_verbose > 0)
        printf ("info: %s: wide_len=%d actual_wide_len=%zu\n",
                __func__, wide_len, actual_wide_len);

      int ret = swprintf (buf, sizeof (buf), L"%.*s%ls",
                          wide_len, mbs, suffix);
      TEST_VERIFY (ret == actual_wide_len + wcslen (suffix));
      TEST_VERIFY (wmemcmp (buf, wide, actual_wide_len) == 0);
      TEST_VERIFY (wcscmp (buf + actual_wide_len, suffix) == 0);

      /* Left-justified string, printed in full.  */
      ret = swprintf (buf, sizeof (buf), L"%-1500.*s%ls",
                      wide_len, mbs, suffix);
      TEST_VERIFY (ret == 1500 + wcslen (suffix));
      TEST_VERIFY (wmemcmp (buf, wide, actual_wide_len) == 0);
      for (size_t i = actual_wide_len; i < 1500; ++i)
        TEST_VERIFY (buf[i] == L' ');
      TEST_VERIFY (wcscmp (buf + 1500, suffix) == 0);

      /* Right-justified string, printed in full.   */
      ret = swprintf (buf, sizeof (buf), L"%1500.*s%ls",
                      wide_len, mbs, suffix);
      TEST_VERIFY (ret == 1500 + wcslen (suffix));
      size_t padding = 1500 - actual_wide_len;
      for (size_t i = 0; i < padding; ++i)
        TEST_VERIFY (buf[i] == L' ');
      TEST_VERIFY (wmemcmp (buf + padding, wide, actual_wide_len) == 0);
      TEST_VERIFY (wcscmp (buf + 1500, suffix) == 0);
    }
}

static int
do_test (void)
{
  /* This test only covers UTF-8 as a multi-byte character set.  A
     locale with a multi-byte character set with shift state would be
     a relevant test target as well, but glibc currently does not ship
     such a locale.  */
  TEST_VERIFY (setlocale (LC_CTYPE, "de_DE.UTF-8") != NULL);

  test_mbs_result ();
  test_wide_result ();

  char *mbs;
  wchar_t *wide;
  size_t *length;
  make_random_string (&mbs, &wide, &length);
  TEST_VERIFY (strlen (mbs) == length[WIDE_STRING_LENGTH]);
  if (test_verbose > 0)
    printf ("info: long multi-byte string contains %zu characters\n",
            length[WIDE_STRING_LENGTH]);
  test_mbs_long (mbs, wide, length);
  test_wide_long (mbs, wide, length);
  free (mbs);
  free (wide);
  free (length);

  return 0;
}

#include <support/test-driver.c>
