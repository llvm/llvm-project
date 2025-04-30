/* Test compilation of truncated regular expressions.
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

/* This test constructs various patterns in an attempt to trigger
   over-reading the regular expression compiler, such as bug
   23578.  */

#include <array_length.h>
#include <errno.h>
#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/next_to_fault.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <wchar.h>

/* Locales to test.  */
static const char locales[][17] =
  {
    "C",
    "en_US.UTF-8",
    "de_DE.ISO-8859-1",
  };

/* Syntax options.  Will be combined with other flags.  */
static const reg_syntax_t syntaxes[] =
  {
    RE_SYNTAX_EMACS,
    RE_SYNTAX_AWK,
    RE_SYNTAX_GNU_AWK,
    RE_SYNTAX_POSIX_AWK,
    RE_SYNTAX_GREP,
    RE_SYNTAX_EGREP,
    RE_SYNTAX_POSIX_EGREP,
    RE_SYNTAX_POSIX_BASIC,
    RE_SYNTAX_POSIX_EXTENDED,
    RE_SYNTAX_POSIX_MINIMAL_EXTENDED,
  };

/* Trailing characters placed after the initial character.  */
static const char trailing_strings[][4] =
  {
    "",
    "[",
    "\\",
    "[\\",
    "(",
    "(\\",
    "\\(",
  };

static int
do_test (void)
{
  /* Staging buffer for the constructed regular expression.  */
  char buffer[16];

  /* Allocation used to detect over-reading by the regular expression
     compiler.  */
  struct support_next_to_fault ntf
    = support_next_to_fault_allocate (sizeof (buffer));

  /* Arbitrary Unicode codepoint at which we stop generating
     characters.  We do not probe the whole range because that would
     take too long due to combinatorical exploision as the result of
     combination with other flags.  */
  static const wchar_t last_character = 0xfff;

  for (size_t locale_idx = 0; locale_idx < array_length (locales);
       ++ locale_idx)
    {
      if (setlocale (LC_ALL, locales[locale_idx]) == NULL)
        {
          support_record_failure ();
          printf ("error: setlocale (\"%s\"): %m", locales[locale_idx]);
          continue;
        }
      if (test_verbose > 0)
        printf ("info: testing locale \"%s\"\n", locales[locale_idx]);

      for (wchar_t wc = 0; wc <= last_character; ++wc)
        {
          char *after_wc;
          if (wc == 0)
            {
              /* wcrtomb treats L'\0' in a special way.  */
              *buffer = '\0';
              after_wc = &buffer[1];
            }
          else
            {
              mbstate_t ps = { };
              size_t ret = wcrtomb (buffer, wc, &ps);
              if (ret == (size_t) -1)
                {
                  /* EILSEQ means that the target character set
                     cannot encode the character.  */
                  if (errno != EILSEQ)
                    {
                      support_record_failure ();
                      printf ("error: wcrtomb (0x%x) failed: %m\n",
                              (unsigned) wc);
                    }
                  continue;
                }
              TEST_VERIFY_EXIT (ret != 0);
              after_wc = &buffer[ret];
            }

          for (size_t trailing_idx = 0;
               trailing_idx < array_length (trailing_strings);
               ++trailing_idx)
            {
              char *after_trailing
                = stpcpy (after_wc, trailing_strings[trailing_idx]);

              for (int do_nul = 0; do_nul < 2; ++do_nul)
                {
                  char *after_nul;
                  if (do_nul)
                    {
                      *after_trailing = '\0';
                      after_nul = &after_trailing[1];
                    }
                  else
                    after_nul = after_trailing;

                  size_t length = after_nul - buffer;

                  /* Make sure that the faulting region starts
                     after the used portion of the buffer.  */
                  char *ntf_start = ntf.buffer + sizeof (buffer) - length;
                  memcpy (ntf_start, buffer, length);

                  for (const reg_syntax_t *psyntax = syntaxes;
                       psyntax < array_end (syntaxes); ++psyntax)
                    for (int do_icase = 0; do_icase < 2; ++do_icase)
                      {
                        re_syntax_options = *psyntax;
                        if (do_icase)
                          re_syntax_options |= RE_ICASE;

                        regex_t reg;
                        memset (&reg, 0, sizeof (reg));
                        const char *msg = re_compile_pattern
                          (ntf_start, length, &reg);
                        if (msg != NULL)
                          {
                            if (test_verbose > 0)
                              {
                                char *quoted = support_quote_blob
                                  (buffer, length);
                                printf ("info: compilation failed for pattern"
                                        " \"%s\", syntax 0x%lx: %s\n",
                                        quoted, re_syntax_options, msg);
                                free (quoted);
                              }
                          }
                        else
                          regfree (&reg);
                      }
                }
            }
        }
    }

  support_next_to_fault_free (&ntf);

  return 0;
}

#include <support/test-driver.c>
