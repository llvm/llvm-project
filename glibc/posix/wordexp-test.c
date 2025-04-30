/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <wordexp.h>
#include <stdio.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include <libc-pointer-arith.h>
#include <array_length.h>
#include <support/xunistd.h>
#include <support/check.h>
#include <support/next_to_fault.h>

#define IFS " \n\t"

struct test_case_struct
{
  int retval;
  const char *env;
  const char *words;
  int flags;
  size_t wordc;
  const char *wordv[10];
  const char *ifs;
} static test_case[] =
  {
    /* Simple word- and field-splitting */
    { 0, NULL, "one", 0, 1, { "one", }, IFS },
    { 0, NULL, "one two", 0, 2, { "one", "two", }, IFS },
    { 0, NULL, "one two three", 0, 3, { "one", "two", "three", }, IFS },
    { 0, NULL, " \tfoo\t\tbar ", 0, 2, { "foo", "bar", }, IFS },
    { 0, NULL, "red , white blue", 0, 4, { "red", ",", "white", "blue", }, " ," },
    { 0, NULL, "one two three", 0, 3, { "one", "two", "three", }, "" },
    { 0, NULL, "one \"two three\"", 0, 2, { "one", "two three", }, IFS },
    { 0, NULL, "one \"two three\"", 0, 2, { "one", "two three", }, "" },
    { 0, "two three", "one \"$var\"", 0, 2, { "one", "two three", }, IFS },
    { 0, "two three", "one $var", 0, 3, { "one", "two", "three", }, IFS },
    { 0, "two three", "one \"$var\"", 0, 2, { "one", "two three", }, "" },
    { 0, "two three", "one $var", 0, 2, { "one", "two three", }, "" },

    /* The non-whitespace IFS char at the end delimits the second field
     * but does NOT start a new field. */
    { 0, ":abc:", "$var", 0, 2, { "", "abc", }, ":" },

    { 0, NULL, "$(echo :abc:)", 0, 2, { "", "abc", }, ":" },
    { 0, NULL, "$(echo :abc:\\ )", 0, 2, { "", "abc", }, ": " },
    { 0, NULL, "$(echo :abc\\ )", 0, 2, { "", "abc", }, ": " },
    { 0, ":abc:", "$(echo $var)", 0, 2, { "", "abc", }, ":" },
    { 0, NULL, ":abc:", 0, 1, { ":abc:", }, ":" },
    { 0, NULL, "$(echo :abc:)def", 0, 3, { "", "abc", "def", },
      ":" },
    { 0, NULL, "$(echo abc:de)f", 0, 2, { "abc", "def", }, ":" },
    { 0, NULL, "$(echo abc:de)f:ghi", 0, 2, { "abc", "def:ghi", },
      ":" },
    { 0, NULL, "abc:d$(echo ef:ghi)", 0, 2, { "abc:def", "ghi", },
      ":" },
    { 0, "abc:", "$var$(echo def:ghi)", 0, 3, { "abc", "def",
							  "ghi", }, ":" },
    { 0, "abc:d", "$var$(echo ef:ghi)", 0, 3, { "abc", "def",
							  "ghi", }, ":" },
    { 0, "def:ghi", "$(echo abc:)$var", 0, 3, { "abc", "def",
							  "ghi", }, ":" },
    { 0, "ef:ghi", "$(echo abc:d)$var", 0, 3, { "abc", "def",
							  "ghi", }, ":" },

    /* Simple parameter expansion */
    { 0, "foo", "${var}", 0, 1, { "foo", }, IFS },
    { 0, "foo", "$var", 0, 1, { "foo", }, IFS },
    { 0, "foo", "\\\"$var\\\"", 0, 1, { "\"foo\"", }, IFS },
    { 0, "foo", "%$var%", 0, 1, { "%foo%", }, IFS },
    { 0, "foo", "-$var-", 0, 1, { "-foo-", }, IFS },

    /* Simple quote removal */
    { 0, NULL, "\"quoted\"", 0, 1, { "quoted", }, IFS },
    { 0, "foo", "\"$var\"\"$var\"", 0, 1, { "foofoo", }, IFS },
    { 0, NULL, "'singly-quoted'", 0, 1, { "singly-quoted", }, IFS },
    { 0, NULL, "contin\\\nuation", 0, 1, { "continuation", }, IFS },
    { 0, NULL, "explicit ''", 0, 2, { "explicit", "", }, IFS },
    { 0, NULL, "explicit \"\"", 0, 2, { "explicit", "", }, IFS },
    { 0, NULL, "explicit ``", 0, 1, { "explicit", }, IFS },

    /* Simple command substitution */
    { 0, NULL, "$(echo hello)", 0, 1, { "hello", }, IFS },
    { 0, NULL, "$( (echo hello) )", 0, 1, { "hello", }, IFS },
    { 0, NULL, "$((echo hello);(echo there))", 0, 2, { "hello", "there", }, IFS },
    { 0, NULL, "`echo one two`", 0, 2, { "one", "two", }, IFS },
    { 0, NULL, "$(echo ')')", 0, 1, { ")" }, IFS },
    { 0, NULL, "$(echo hello; echo)", 0, 1, { "hello", }, IFS },
    { 0, NULL, "a$(echo b)c", 0, 1, { "abc", }, IFS },

    /* Simple arithmetic expansion */
    { 0, NULL, "$((1 + 1))", 0, 1, { "2", }, IFS },
    { 0, NULL, "$((2-3))", 0, 1, { "-1", }, IFS },
    { 0, NULL, "$((-1))", 0, 1, { "-1", }, IFS },
    { 0, NULL, "$[50+20]", 0, 1, { "70", }, IFS },
    { 0, NULL, "$(((2+3)*(4+5)))", 0, 1, { "45", }, IFS },
    { 0, NULL, "$((010))", 0, 1, { "8" }, IFS },
    { 0, NULL, "$((0x10))", 0, 1, { "16" }, IFS },
    { 0, NULL, "$((010+0x10))", 0, 1, { "24" }, IFS },
    { 0, NULL, "$((-010+0x10))", 0, 1, { "8" }, IFS },
    { 0, NULL, "$((-0x10+010))", 0, 1, { "-8" }, IFS },

    /* Advanced parameter expansion */
    { 0, NULL, "${var:-bar}", 0, 1, { "bar", }, IFS },
    { 0, NULL, "${var-bar}", 0, 1, { "bar", }, IFS },
    { 0, "", "${var:-bar}", 0, 1, { "bar", }, IFS },
    { 0, "foo", "${var:-bar}", 0, 1, { "foo", }, IFS },
    { 0, "", "${var-bar}", 0, 0, { NULL, }, IFS },
    { 0, NULL, "${var:=bar}", 0, 1, { "bar", }, IFS },
    { 0, NULL, "${var=bar}", 0, 1, { "bar", }, IFS },
    { 0, "", "${var:=bar}", 0, 1, { "bar", }, IFS },
    { 0, "foo", "${var:=bar}", 0, 1, { "foo", }, IFS },
    { 0, "", "${var=bar}", 0, 0, { NULL, }, IFS },
    { 0, "foo", "${var:?bar}", 0, 1, { "foo", }, IFS },
    { 0, NULL, "${var:+bar}", 0, 0, { NULL, }, IFS },
    { 0, NULL, "${var+bar}", 0, 0, { NULL, }, IFS },
    { 0, "", "${var:+bar}", 0, 0, { NULL, }, IFS },
    { 0, "foo", "${var:+bar}", 0, 1, { "bar", }, IFS },
    { 0, "", "${var+bar}", 0, 1, { "bar", }, IFS },
    { 0, "12345", "${#var}", 0, 1, { "5", }, IFS },
    { 0, NULL, "${var:-'}'}", 0, 1, { "}", }, IFS },
    { 0, NULL, "${var-}", 0, 0, { NULL }, IFS },

    { 0, "pizza", "${var#${var}}", 0, 0, { NULL }, IFS },
    { 0, "pepperoni", "${var%$(echo oni)}", 0, 1, { "pepper" }, IFS },
    { 0, "6pack", "${var#$((6))}", 0, 1, { "pack" }, IFS },
    { 0, "b*witched", "${var##b*}", 0, 0, { NULL }, IFS },
    { 0, "b*witched", "${var##\"b*\"}", 0, 1, { "witched" }, IFS },
    { 0, "banana", "${var%na*}", 0, 1, { "bana", }, IFS },
    { 0, "banana", "${var%%na*}", 0, 1, { "ba", }, IFS },
    { 0, "borabora-island", "${var#*bora}", 0, 1, { "bora-island", }, IFS },
    { 0, "borabora-island", "${var##*bora}", 0, 1, { "-island", }, IFS },
    { 0, "coconut", "${var##\\*co}", 0, 1, { "coconut", }, IFS },
    { 0, "100%", "${var%0%}", 0, 1, { "10" }, IFS },

    /* Pathname expansion */
    { 0, NULL, "???", 0, 2, { "one", "two", }, IFS },
    { 0, NULL, "[ot]??", 0, 2, { "one", "two", }, IFS },
    { 0, NULL, "t*", 0, 2, { "three", "two", }, IFS },
    { 0, NULL, "\"t\"*", 0, 2, { "three", "two", }, IFS },

    /* Nested constructs */
    { 0, "one two", "$var", 0, 2, { "one", "two", }, IFS },
    { 0, "one two three", "$var", 0, 3, { "one", "two", "three", }, IFS },
    { 0, " \tfoo\t\tbar ", "$var", 0, 2, { "foo", "bar", }, IFS },
    { 0, "  red  , white blue", "$var", 0, 3, { "red", "white", "blue", }, ", \n\t" },
    { 0, "  red  , white blue", "\"$var\"", 0, 1, { "  red  , white blue", }, ", \n\t" },
    { 0, NULL, "\"$(echo hello there)\"", 0, 1, { "hello there", }, IFS },
    { 0, NULL, "\"$(echo \"hello there\")\"", 0, 1, { "hello there", }, IFS },
    { 0, NULL, "${var=one two} \"$var\"", 0, 3, { "one", "two", "one two", }, IFS },
    { 0, "1", "$(( $(echo 3)+$var ))", 0, 1, { "4", }, IFS },
    { 0, NULL, "\"$(echo \"*\")\"", 0, 1, { "*", }, IFS },
    { 0, NULL, "\"a\n\n$(echo)b\"", 0, 1, { "a\n\nb", }, IFS },
    { 0, "foo", "*$var*", 0, 1, { "*foo*", }, IFS },
    { 0, "o thr", "*$var*", 0, 2, { "two", "three" }, IFS },

    /* Different IFS values */
    { 0, "a b\tc\nd  ", "$var", 0, 4, { "a", "b", "c", "d" }, NULL /* unset */ },
    { 0, "a b\tc d  ", "$var", 0, 1, { "a b\tc d  " }, "" /* `null' */ },
    { 0, "a,b c\n, d", "$var", 0, 3, { "a", "b c", " d" }, "\t\n," },

    /* Other things that should succeed */
    { 0, NULL, "\\*\"|&;<>\"\\(\\)\\{\\}", 0, 1, { "*|&;<>(){}", }, IFS },
    { 0, "???", "$var", 0, 1, { "???", }, IFS },
    { 0, NULL, "$var", 0, 0, { NULL, }, IFS },
    { 0, NULL, "\"\\n\"", 0, 1, { "\\n", }, IFS },
    { 0, NULL, "", 0, 0, { NULL, }, IFS },
    { 0, NULL, "${1234567890123456789012}", 0, 0, { NULL, }, IFS },

    /* Flags not already covered (testit() has special handling for these) */
    { 0, NULL, "one two", WRDE_DOOFFS, 2, { "one", "two", }, IFS },
    { 0, NULL, "appended", WRDE_APPEND, 3, { "pre1", "pre2", "appended", }, IFS },
    { 0, NULL, "appended", WRDE_DOOFFS|WRDE_APPEND, 3, { "pre1", "pre2", "appended", }, IFS },

    /* Things that should fail */
    { WRDE_BADCHAR, NULL, "new\nline", 0, 0, { NULL, }, "" /* \n not IFS */ },
    { WRDE_BADCHAR, NULL, "pipe|symbol", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "&ampersand", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "semi;colon", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "<greater", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "less>", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "(open-paren", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "close-paren)", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "{open-brace", 0, 0, { NULL, }, IFS },
    { WRDE_BADCHAR, NULL, "close-brace}", 0, 0, { NULL, }, IFS },
    { WRDE_BADVAL, NULL, "$var", WRDE_UNDEF, 0, { NULL, }, IFS },
    { WRDE_BADVAL, NULL, "$9", WRDE_UNDEF, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "$[50+20))", 0, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "${%%noparam}", 0, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "${missing-brace", 0, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "$(for i in)", 0, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "$((2+))", 0, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "`", 0, 0, { NULL, }, IFS },
    { WRDE_SYNTAX, NULL, "$((010+4+))", 0, 0, { NULL }, IFS },

    { WRDE_SYNTAX, NULL, "`\\", 0, 0, { NULL, }, IFS },     /* BZ 18042  */
    { WRDE_SYNTAX, NULL, "${", 0, 0, { NULL, }, IFS },      /* BZ 18043  */
    { WRDE_SYNTAX, NULL, "L${a:", 0, 0, { NULL, }, IFS },   /* BZ 18043#c4  */
  };

static int testit (struct test_case_struct *tc);
static int tests;

static void
command_line_test (const char *words)
{
  wordexp_t we;
  int i;
  int retval = wordexp (words, &we, 0);
  printf ("info: wordexp returned %d\n", retval);
  for (i = 0; i < we.we_wordc; i++)
    printf ("info: we_wordv[%d] = \"%s\"\n", i, we.we_wordv[i]);
}

static int
do_test (int argc, char *argv[])
{
  const char *globfile[] = { "one", "two", "three" };
  char tmpdir[32];
  struct passwd *pw;
  const char *cwd;
  int test;
  struct test_case_struct ts;

  if (argc > 1)
    {
      command_line_test (argv[1]);
      return 0;
    }

  cwd = getcwd (NULL, 0);

  /* Set up arena for pathname expansion */
  tmpnam (tmpdir);
  xmkdir (tmpdir, S_IRWXU);
  TEST_VERIFY_EXIT (chdir (tmpdir) == 0);

  for (int i = 0; i < array_length (globfile); ++i)
    {
      int fd = xopen (globfile[i], O_WRONLY|O_CREAT|O_TRUNC,
		      S_IRUSR | S_IWUSR);
      xclose (fd);
    }

  for (test = 0; test < array_length (test_case); test++)
    TEST_COMPARE (testit (&test_case[test]), 0);

  /* Tilde-expansion tests. */
  pw = getpwnam ("root");
  if (pw != NULL)
    {
      ts.retval = 0;
      ts.env = NULL;
      ts.words = "~root ";
      ts.flags = 0;
      ts.wordc = 1;
      ts.wordv[0] = pw->pw_dir;
      ts.ifs = IFS;

      TEST_COMPARE (testit (&ts), 0);

      ts.retval = 0;
      ts.env = pw->pw_dir;
      ts.words = "${var#~root}x";
      ts.flags = 0;
      ts.wordc = 1;
      ts.wordv[0] = "x";
      ts.ifs = IFS;

      TEST_COMPARE (testit (&ts), 0);
    }

  /* "~" expands to value of $HOME when HOME is set */

  setenv ("HOME", "/dummy/home", 1);
  ts.retval = 0;
  ts.env = NULL;
  ts.words = "~ ~/foo";
  ts.flags = 0;
  ts.wordc = 2;
  ts.wordv[0] = "/dummy/home";
  ts.wordv[1] = "/dummy/home/foo";
  ts.ifs = IFS;

  TEST_COMPARE (testit (&ts), 0);

  /* "~" expands to home dir from passwd file if HOME is not set */

  pw = getpwuid (getuid ());
  if (pw != NULL)
    {
      unsetenv ("HOME");
      ts.retval = 0;
      ts.env = NULL;
      ts.words = "~";
      ts.flags = 0;
      ts.wordc = 1;
      ts.wordv[0] = pw->pw_dir;
      ts.ifs = IFS;

      TEST_COMPARE (testit (&ts), 0);
    }

  puts ("tests completed, now cleaning up");

  /* Clean up */
  for (int i = 0; i < array_length (globfile); ++i)
    remove (globfile[i]);

  if (cwd == NULL)
    cwd = "..";

  chdir (cwd);
  rmdir (tmpdir);

  return 0;
}

struct support_next_to_fault
at_page_end (const char *words)
{
  const size_t words_size = strlen (words) + 1;
  struct support_next_to_fault ntf
    = support_next_to_fault_allocate (words_size);

  /* Includes terminating NUL.  */
  memcpy (ntf.buffer, words, words_size);

  return ntf;
}

static int
testit (struct test_case_struct *tc)
{
  int retval;
  wordexp_t we, sav_we;
  char *dummy;
  int bzzzt = 0;
  int start_offs = 0;
  int i;

  if (tc->env)
    setenv ("var", tc->env, 1);
  else
    unsetenv ("var");

  if (tc->ifs)
    setenv ("IFS", tc->ifs, 1);
  else
    unsetenv ("IFS");

  sav_we.we_wordc = 99;
  sav_we.we_wordv = &dummy;
  sav_we.we_offs = 3;
  we = sav_we;

  printf ("info: test %d (%s): ", ++tests, tc->words);
  fflush (NULL);
  struct support_next_to_fault words = at_page_end (tc->words);

  if (tc->flags & WRDE_APPEND)
    {
      /* initial wordexp() call, to be appended to */
      if (wordexp ("pre1 pre2", &we, tc->flags & ~WRDE_APPEND) != 0)
        {
	  printf ("info: FAILED setup\n");
	  return 1;
	}
    }
  retval = wordexp (words.buffer, &we, tc->flags);

  if (tc->flags & WRDE_DOOFFS)
      start_offs = sav_we.we_offs;

  if (retval != tc->retval || (retval == 0 && we.we_wordc != tc->wordc))
    bzzzt = 1;
  else if (retval == 0)
    {
      for (i = 0; i < start_offs; ++i)
	if (we.we_wordv[i] != NULL)
	  {
	    bzzzt = 1;
	    break;
	  }

      for (i = 0; i < we.we_wordc; ++i)
	if (we.we_wordv[i+start_offs] == NULL
	    || strcmp (tc->wordv[i], we.we_wordv[i+start_offs]) != 0)
	  {
	    bzzzt = 1;
	    break;
	  }
    }

  if (bzzzt)
    {
      printf ("FAILED\n");
      printf ("info: Test words: <%s>, need retval %d, wordc %Zd\n",
	      tc->words, tc->retval, tc->wordc);
      if (start_offs != 0)
	printf ("(preceded by %d NULLs)\n", start_offs);
      printf ("Got retval %d, wordc %Zd: ", retval, we.we_wordc);
      if (retval == 0 || retval == WRDE_NOSPACE)
	{
	  for (i = 0; i < we.we_wordc + start_offs; ++i)
	    if (we.we_wordv[i] == NULL)
	      printf ("NULL ");
	    else
	      printf ("<%s> ", we.we_wordv[i]);
	}
      printf ("\n");
    }
  else if (retval != 0 && retval != WRDE_NOSPACE
	   && (we.we_wordc != sav_we.we_wordc
	       || we.we_wordv != sav_we.we_wordv
	       || we.we_offs != sav_we.we_offs))
    {
      bzzzt = 1;
      printf ("FAILED to restore wordexp_t members\n");
    }
  else
    printf ("OK\n");

  if (retval == 0 || retval == WRDE_NOSPACE)
    wordfree (&we);

  support_next_to_fault_free (&words);

  fflush (NULL);
  return bzzzt;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
