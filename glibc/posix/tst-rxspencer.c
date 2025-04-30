/* Regular expression tests.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <sys/types.h>
#include <mcheck.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <getopt.h>

static void
replace_special_chars (char *str)
{
  for (; (str = strpbrk (str, "NTSZ")) != NULL; ++str)
    switch (*str)
      {
      case 'N': *str = '\n'; break;
      case 'T': *str = '\t'; break;
      case 'S': *str = ' '; break;
      case 'Z': *str = '\0'; break;
      }
}

static void
glibc_re_syntax (char *str)
{
  char *p, *end = strchr (str, '\0') + 1;

  /* Replace [[:<:]] with \< and [[:>:]] with \>.  */
  for (p = str; (p = strstr (p, "[[:")) != NULL; )
    if ((p[3] == '<' || p[3] == '>') && strncmp (p + 4, ":]]", 3) == 0)
      {
        p[0] = '\\';
        p[1] = p[3];
        memmove (p + 2, p + 7, end - p - 7);
        end -= 5;
        p += 2;
      }
    else
      p += 3;
}

static char *
mb_replace (char *dst, const char c)
{
  switch (c)
    {
    /* Replace a with \'a and A with \'A.  */
    case 'a':
      *dst++ = '\xc3';
      *dst++ = '\xa1';
      break;
    case 'A':
      *dst++ = '\xc3';
      *dst++ = '\x81';
      break;
    /* Replace b with \v{c} and B with \v{C}.  */
    case 'b':
      *dst++ = '\xc4';
      *dst++ = '\x8d';
      break;
    case 'B':
      *dst++ = '\xc4';
      *dst++ = '\x8c';
      break;
    /* Replace c with \v{d} and C with \v{D}.  */
    case 'c':
      *dst++ = '\xc4';
      *dst++ = '\x8f';
      break;
    case 'C':
      *dst++ = '\xc4';
      *dst++ = '\x8e';
      break;
    /* Replace d with \'e and D with \'E.  */
    case 'd':
      *dst++ = '\xc3';
      *dst++ = '\xa9';
      break;
    case 'D':
      *dst++ = '\xc3';
      *dst++ = '\x89';
      break;
    }
  return dst;
}

static char *
mb_frob_string (const char *str, const char *letters)
{
  char *ret, *dst;
  const char *src;

  if (str == NULL)
    return NULL;

  ret = malloc (2 * strlen (str) + 1);
  if (ret == NULL)
    return NULL;

  for (src = str, dst = ret; *src; ++src)
    if (strchr (letters, *src))
      dst = mb_replace (dst, *src);
    else
      *dst++ = *src;
  *dst = '\0';
  return ret;
}

/* Like mb_frob_string, but don't replace anything between
   [: and :], [. and .] or [= and =] or characters escaped
   with a backslash.  */

static char *
mb_frob_pattern (const char *str, const char *letters)
{
  char *ret, *dst;
  const char *src;
  int in_class = 0, escaped = 0;

  if (str == NULL)
    return NULL;

  ret = malloc (2 * strlen (str) + 1);
  if (ret == NULL)
    return NULL;

  for (src = str, dst = ret; *src; ++src)
    if (*src == '\\')
      {
	escaped ^= 1;
	*dst++ = *src;
      }
    else if (escaped)
      {
	escaped = 0;
	*dst++ = *src;
	continue;
      }
    else if (!in_class && strchr (letters, *src))
      dst = mb_replace (dst, *src);
    else
      {
	if (!in_class && *src == '[' && strchr (":.=", src[1]))
	  in_class = 1;
	else if (in_class && *src == ']' && strchr (":.=", src[-1]))
	  in_class = 0;
	*dst++ = *src;
      }
  *dst = '\0';
  return ret;
}

static int
check_match (regmatch_t *rm, int idx, const char *string,
	     const char *match, const char *fail)
{
  if (match[0] == '-' && match[1] == '\0')
    {
      if (rm[idx].rm_so == -1 && rm[idx].rm_eo == -1)
	return 0;
      printf ("%s rm[%d] unexpectedly matched\n", fail, idx);
      return 1;
    }

  if (rm[idx].rm_so == -1 || rm[idx].rm_eo == -1)
    {
      printf ("%s rm[%d] unexpectedly did not match\n", fail, idx);
      return 1;
    }

  if (match[0] == '@')
    {
      if (rm[idx].rm_so != rm[idx].rm_eo)
	{
	  printf ("%s rm[%d] not empty\n", fail, idx);
	  return 1;
	}

      if (strncmp (string + rm[idx].rm_so, match + 1, strlen (match + 1) ?: 1))
	{
	  printf ("%s rm[%d] not matching %s\n", fail, idx, match);
	  return 1;
	}
      return 0;
    }

  if (rm[idx].rm_eo - rm[idx].rm_so != strlen (match)
      || strncmp (string + rm[idx].rm_so, match,
		  rm[idx].rm_eo - rm[idx].rm_so))
    {
      printf ("%s rm[%d] not matching %s\n", fail, idx, match);
      return 1;
    }

  return 0;
}

static int
test (const char *pattern, int cflags, const char *string, int eflags,
      char *expect, char *matches, const char *fail)
{
  regex_t re;
  regmatch_t rm[10];
  int n, ret = 0;

  n = regcomp (&re, pattern, cflags);
  if (n != 0)
    {
      char buf[500];
      if (eflags == -1)
	{
	  static struct { reg_errcode_t code; const char *name; } codes []
#define C(x) { REG_##x, #x }
	    = { C(NOERROR), C(NOMATCH), C(BADPAT), C(ECOLLATE),
		C(ECTYPE), C(EESCAPE), C(ESUBREG), C(EBRACK),
		C(EPAREN), C(EBRACE), C(BADBR), C(ERANGE),
		C(ESPACE), C(BADRPT) };

	  for (int i = 0; i < sizeof (codes) / sizeof (codes[0]); ++i)
	    if (n == codes[i].code)
	      {
		if (strcmp (string, codes[i].name))
		  {
		    printf ("%s regcomp returned REG_%s (expected REG_%s)\n",
			    fail, codes[i].name, string);
		    return 1;
		  }
	        return 0;
	      }

	  printf ("%s regcomp return value REG_%d\n", fail, n);
	  return 1;
	}

      regerror (n, &re, buf, sizeof (buf));
      printf ("%s regcomp failed: %s\n", fail, buf);
      return 1;
    }

  if (eflags == -1)
    {
      regfree (&re);

      /* The test case file assumes something only guaranteed by the
	 rxspencer regex implementation.  Namely that for empty
	 expressions regcomp() return REG_EMPTY.  This is not the case
	 for us and so we ignore this error.  */
      if (strcmp (string, "EMPTY") == 0)
	return 0;

      printf ("%s regcomp unexpectedly succeeded\n", fail);
      return 1;
    }

  if (regexec (&re, string, 10, rm, eflags))
    {
      regfree (&re);
      if (expect == NULL)
	return 0;
      printf ("%s regexec failed\n", fail);
      return 1;
    }

  regfree (&re);

  if (expect == NULL)
    {
      printf ("%s regexec unexpectedly succeeded\n", fail);
      return 1;
    }

  if (cflags & REG_NOSUB)
    return 0;

  ret = check_match (rm, 0, string, expect, fail);
  if (matches == NULL)
    return ret;

  for (n = 1; ret == 0 && n < 10; ++n)
    {
      char *p = NULL;

      if (matches)
	{
	  p = strchr (matches, ',');
	  if (p != NULL)
	    *p = '\0';
	}
      ret = check_match (rm, n, string, matches ?: "-", fail);
      if (p)
	{
	  *p = ',';
	  matches = p + 1;
	}
      else
	matches = NULL;
    }

  return ret;
}

static int
mb_test (const char *pattern, int cflags, const char *string, int eflags,
	 char *expect, const char *matches, const char *letters,
	 const char *fail)
{
  char *pattern_mb = mb_frob_pattern (pattern, letters);
  const char *string_mb
    = eflags == -1 ? string : mb_frob_string (string, letters);
  char *expect_mb = mb_frob_string (expect, letters);
  char *matches_mb = mb_frob_string (matches, letters);
  int ret = 0;

  if (!pattern_mb || !string_mb
      || (expect && !expect_mb) || (matches && !matches_mb))
    {
      printf ("%s %m", fail);
      ret = 1;
    }
  else
    ret = test (pattern_mb, cflags, string_mb, eflags, expect_mb,
		matches_mb, fail);

  free (matches_mb);
  free (expect_mb);
  if (string_mb != string)
    free ((char *) string_mb);
  free (pattern_mb);
  return ret;
}

static int
mb_tests (const char *pattern, int cflags, const char *string, int eflags,
	  char *expect, const char *matches)
{
  int ret = 0;
  int i;
  char letters[9], fail[20];

  /* The tests aren't supposed to work with xdigit, since a-dA-D are
     hex digits while \'a \'A \v{c}\v{C}\v{d}\v{D}\'e \'E are not.  */
  if (strstr (pattern, "[:xdigit:]"))
    return 0;

  /* XXX: regex ATM handles only single byte equivalence classes.  */
  if (strstr (pattern, "[[=b=]]"))
    return 0;

  for (i = 1; i < 16; ++i)
    {
      char *p = letters;
      if (i & 1)
	{
	  if (!strchr (pattern, 'a') && !strchr (string, 'a')
	      && !strchr (pattern, 'A') && !strchr (string, 'A'))
	    continue;
	  *p++ = 'a', *p++ = 'A';
	}
      if (i & 2)
	{
	  if (!strchr (pattern, 'b') && !strchr (string, 'b')
	      && !strchr (pattern, 'B') && !strchr (string, 'B'))
	    continue;
	  *p++ = 'b', *p++ = 'B';
	}
      if (i & 4)
	{
	  if (!strchr (pattern, 'c') && !strchr (string, 'c')
	      && !strchr (pattern, 'C') && !strchr (string, 'C'))
	    continue;
	  *p++ = 'c', *p++ = 'C';
	}
      if (i & 8)
	{
	  if (!strchr (pattern, 'd') && !strchr (string, 'd')
	      && !strchr (pattern, 'D') && !strchr (string, 'D'))
	    continue;
	  *p++ = 'd', *p++ = 'D';
	}
      *p++ = '\0';
      sprintf (fail, "UTF-8 %s FAIL", letters);
      ret |= mb_test (pattern, cflags, string, eflags, expect, matches,
		      letters, fail);
    }
  return ret;
}

int
main (int argc, char **argv)
{
  int ret = 0;
  char *line = NULL;
  size_t line_len = 0;
  ssize_t len;
  FILE *f;
  static int test_utf8 = 0;
  static const struct option options[] =
    {
      {"utf8",	no_argument,	&test_utf8,	1},
      {NULL,	0,		NULL,		0 }
    };

  mtrace ();

  while (getopt_long (argc, argv, "", options, NULL) >= 0);

  if (optind + 1 != argc)
    {
      fprintf (stderr, "Missing test filename\n");
      return 1;
    }

  f = fopen (argv[optind], "r");
  if (f == NULL)
    {
      fprintf (stderr, "Couldn't open %s\n", argv[optind]);
      return 1;
    }

  while ((len = getline (&line, &line_len, f)) > 0)
    {
      char *pattern, *flagstr, *string, *expect, *matches, *p;
      int cflags = REG_EXTENDED, eflags = 0, try_bre_ere = 0;

      if (line[len - 1] == '\n')
        line[len - 1] = '\0';

      /* Skip comments and empty lines.  */
      if (*line == '#' || *line == '\0')
	continue;

      puts (line);
      fflush (stdout);

      pattern = strtok (line, "\t");
      if (pattern == NULL)
        continue;

      if (strcmp (pattern, "\"\"") == 0)
	pattern += 2;

      flagstr = strtok (NULL, "\t");
      if (flagstr == NULL)
        continue;

      string = strtok (NULL, "\t");
      if (string == NULL)
        continue;

      if (strcmp (string, "\"\"") == 0)
	string += 2;

      for (p = flagstr; *p; ++p)
	switch (*p)
	  {
	  case '-':
	    break;
	  case 'b':
	    cflags &= ~REG_EXTENDED;
	    break;
	  case '&':
	    try_bre_ere = 1;
	    break;
	  case 'C':
	    eflags = -1;
	    break;
	  case 'i':
	    cflags |= REG_ICASE;
	    break;
	  case 's':
	    cflags |= REG_NOSUB;
	    break;
	  case 'n':
	    cflags |= REG_NEWLINE;
	    break;
	  case '^':
	    eflags |= REG_NOTBOL;
	    break;
	  case '$':
	    eflags |= REG_NOTEOL;
	    break;
	  case 'm':
	  case 'p':
	  case '#':
	    /* Not supported.  */
	    flagstr = NULL;
	    break;
	  }

      if (flagstr == NULL)
	continue;

      replace_special_chars (pattern);
      glibc_re_syntax (pattern);
      if (eflags != -1)
        replace_special_chars (string);

      expect = strtok (NULL, "\t");
      matches = NULL;
      if (expect != NULL)
        {
	  replace_special_chars (expect);
	  matches = strtok (NULL, "\t");
	  if (matches != NULL)
	    replace_special_chars (matches);
        }

      if (setlocale (LC_ALL, "C") == NULL)
	{
	  puts ("setlocale C failed");
	  ret = 1;
	}
      if (test (pattern, cflags, string, eflags, expect, matches, "FAIL")
	  || (try_bre_ere
	      && test (pattern, cflags & ~REG_EXTENDED, string, eflags,
		       expect, matches, "FAIL")))
	ret = 1;
      else if (test_utf8)
	{
	  if (setlocale (LC_ALL, "cs_CZ.UTF-8") == NULL)
	    {
	      puts ("setlocale cs_CZ.UTF-8 failed");
	      ret = 1;
	    }
	  else if (test (pattern, cflags, string, eflags, expect, matches,
			 "UTF-8 FAIL")
		   || (try_bre_ere
		       && test (pattern, cflags & ~REG_EXTENDED, string,
				eflags, expect, matches, "UTF-8 FAIL")))
	    ret = 1;
	  else if (mb_tests (pattern, cflags, string, eflags, expect, matches)
		   || (try_bre_ere
		       && mb_tests (pattern, cflags & ~REG_EXTENDED, string,
				    eflags, expect, matches)))
	    ret = 1;
	}
    }

  free (line);
  fclose (f);
  return ret;
}
