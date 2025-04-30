/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <getopt.h>
#include <iconv.h>
#include <locale.h>
#include <mcheck.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <regex.h>


#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
static clockid_t cl;
static int use_clock;
#endif
static iconv_t cd;
static char *mem;
static char *umem;
static size_t memlen;
static size_t umemlen;
static int timing;

static int test_expr (const char *expr, int expected, int expectedicase);
static int run_test (const char *expr, const char *mem, size_t memlen,
		     int icase, int expected);
static int run_test_backwards (const char *expr, const char *mem,
			       size_t memlen, int icase, int expected);


static int
do_test (void)
{
  const char *file;
  int fd;
  struct stat st;
  int result;
  char *inmem;
  char *outmem;
  size_t inlen;
  size_t outlen;

  mtrace ();

  /* Make the content of the file available in memory.  */
  file = "./tst-regex.input";
  fd = open (file, O_RDONLY);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot open %s", basename (file));

  if (fstat (fd, &st) != 0)
    error (EXIT_FAILURE, errno, "cannot stat %s", basename (file));
  memlen = st.st_size;

  mem = (char *) malloc (memlen + 1);
  if (mem == NULL)
    error (EXIT_FAILURE, errno, "while allocating buffer");

  if ((size_t) read (fd, mem, memlen) != memlen)
    error (EXIT_FAILURE, 0, "cannot read entire file");
  mem[memlen] = '\0';

  close (fd);

  /* We have to convert a few things from UTF-8 to Latin-1.  */
  cd = iconv_open ("ISO-8859-1", "UTF-8");
  if (cd == (iconv_t) -1)
    error (EXIT_FAILURE, errno, "cannot get conversion descriptor");

  /* For the second test we have to convert the file content to Latin-1.
     This cannot grow the data.  */
  umem = (char *) malloc (memlen + 1);
  if (umem == NULL)
    error (EXIT_FAILURE, errno, "while allocating buffer");

  inmem = mem;
  inlen = memlen;
  outmem = umem;
  outlen = memlen;
  iconv (cd, &inmem, &inlen, &outmem, &outlen);
  umemlen = outmem - umem;
  if (inlen != 0)
    error (EXIT_FAILURE, errno, "cannot convert buffer");
  umem[umemlen] = '\0';

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
# if _POSIX_CPUTIME == 0
  if (sysconf (_SC_CPUTIME) < 0)
    use_clock = 0;
  else
# endif
    /* See whether we can use the CPU clock.  */
    use_clock = clock_getcpuclockid (0, &cl) == 0;
#endif

#ifdef DEBUG
  re_set_syntax (RE_DEBUG);
#endif

  /* Run the actual tests.  All tests are run in a single-byte and a
     multi-byte locale.  */
  result = test_expr ("[äáàâéèêíìîñöóòôüúùû]", 4, 4);
  result |= test_expr ("G.ran", 2, 3);
  result |= test_expr ("G.\\{1\\}ran", 2, 3);
  result |= test_expr ("G.*ran", 3, 44);
  result |= test_expr ("[äáàâ]", 0, 0);
  result |= test_expr ("Uddeborg", 2, 2);
  result |= test_expr (".Uddeborg", 2, 2);

  /* Free the resources.  */
  free (umem);
  iconv_close (cd);
  free (mem);

  return result;
}


static int
test_expr (const char *expr, int expected, int expectedicase)
{
  int result;
  char *inmem;
  char *outmem;
  size_t inlen;
  size_t outlen;
  char *uexpr;

  /* First test: search with an UTF-8 locale.  */
  if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
    error (EXIT_FAILURE, 0, "cannot set locale de_DE.UTF-8");

  printf ("\nTest \"%s\" with multi-byte locale\n", expr);
  result = run_test (expr, mem, memlen, 0, expected);
  printf ("\nTest \"%s\" with multi-byte locale, case insensitive\n", expr);
  result |= run_test (expr, mem, memlen, 1, expectedicase);
  printf ("\nTest \"%s\" backwards with multi-byte locale\n", expr);
  result |= run_test_backwards (expr, mem, memlen, 0, expected);
  printf ("\nTest \"%s\" backwards with multi-byte locale, case insensitive\n",
	  expr);
  result |= run_test_backwards (expr, mem, memlen, 1, expectedicase);

  /* Second test: search with an ISO-8859-1 locale.  */
  if (setlocale (LC_ALL, "de_DE.ISO-8859-1") == NULL)
    error (EXIT_FAILURE, 0, "cannot set locale de_DE.ISO-8859-1");

  inmem = (char *) expr;
  inlen = strlen (expr);
  outlen = inlen;
  outmem = uexpr = alloca (outlen + 1);
  memset (outmem, '\0', outlen + 1);
  iconv (cd, &inmem, &inlen, &outmem, &outlen);
  if (inlen != 0)
    error (EXIT_FAILURE, errno, "cannot convert expression");

  /* Run the tests.  */
  printf ("\nTest \"%s\" with 8-bit locale\n", expr);
  result |= run_test (uexpr, umem, umemlen, 0, expected);
  printf ("\nTest \"%s\" with 8-bit locale, case insensitive\n", expr);
  result |= run_test (uexpr, umem, umemlen, 1, expectedicase);
  printf ("\nTest \"%s\" backwards with 8-bit locale\n", expr);
  result |= run_test_backwards (uexpr, umem, umemlen, 0, expected);
  printf ("\nTest \"%s\" backwards with 8-bit locale, case insensitive\n",
	  expr);
  result |= run_test_backwards (uexpr, umem, umemlen, 1, expectedicase);

  return result;
}


static int
run_test (const char *expr, const char *mem, size_t memlen, int icase,
	  int expected)
{
#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
  struct timespec start;
  struct timespec finish;
#endif
  regex_t re;
  int err;
  size_t offset;
  int cnt;

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
  if (use_clock && !timing)
    use_clock = clock_gettime (cl, &start) == 0;
#endif

  err = regcomp (&re, expr, REG_NEWLINE | (icase ? REG_ICASE : 0));
  if (err != REG_NOERROR)
    {
      char buf[200];
      regerror (err, &re, buf, sizeof buf);
      error (EXIT_FAILURE, 0, "cannot compile expression: %s", buf);
    }

  cnt = 0;
  offset = 0;
  assert (mem[memlen] == '\0');
  while (offset < memlen)
    {
      regmatch_t ma[1];
      const char *sp;
      const char *ep;

      err = regexec (&re, mem + offset, 1, ma, 0);
      if (err == REG_NOMATCH)
	break;

      if (err != REG_NOERROR)
	{
	  char buf[200];
	  regerror (err, &re, buf, sizeof buf);
	  error (EXIT_FAILURE, 0, "cannot use expression: %s", buf);
	}

      assert (ma[0].rm_so >= 0);
      sp = mem + offset + ma[0].rm_so;
      while (sp > mem && sp[-1] != '\n')
	--sp;

      ep = mem + offset + ma[0].rm_so;
      while (*ep != '\0' && *ep != '\n')
	++ep;

      printf ("match %d: \"%.*s\"\n", ++cnt, (int) (ep - sp), sp);

      offset = ep + 1 - mem;
    }

  regfree (&re);

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
  if (use_clock && !timing)
    {
      use_clock = clock_gettime (cl, &finish) == 0;
      if (use_clock)
	{
	  if (finish.tv_nsec < start.tv_nsec)
	    {
	      finish.tv_nsec -= start.tv_nsec - 1000000000;
	      finish.tv_sec -= 1 + start.tv_sec;
	    }
	  else
	    {
	      finish.tv_nsec -= start.tv_nsec;
	      finish.tv_sec -= start.tv_sec;
	    }

	  printf ("elapsed time: %jd.%09jd sec\n",
		  (intmax_t) finish.tv_sec, (intmax_t) finish.tv_nsec);
	}
    }

  if (use_clock && timing)
    {
      struct timespec mintime = { .tv_sec = 24 * 60 * 60 };

      for (int i = 0; i < 10; ++i)
	{
	  offset = 0;
	  use_clock = clock_gettime (cl, &start) == 0;

	  if (!use_clock)
	    continue;

	  err = regcomp (&re, expr, REG_NEWLINE | (icase ? REG_ICASE : 0));
	  if (err != REG_NOERROR)
	    continue;

	  while (offset < memlen)
	    {
	      regmatch_t ma[1];

	      err = regexec (&re, mem + offset, 1, ma, 0);
	      if (err != REG_NOERROR)
		break;

	      offset += ma[0].rm_eo;
	    }

	  regfree (&re);

	  use_clock = clock_gettime (cl, &finish) == 0;
	  if (use_clock)
	    {
	      if (finish.tv_nsec < start.tv_nsec)
		{
		  finish.tv_nsec -= start.tv_nsec - 1000000000;
		  finish.tv_sec -= 1 + start.tv_sec;
		}
	      else
		{
		  finish.tv_nsec -= start.tv_nsec;
		  finish.tv_sec -= start.tv_sec;
		}
	      if (finish.tv_sec < mintime.tv_sec
		  || (finish.tv_sec == mintime.tv_sec
		      && finish.tv_nsec < mintime.tv_nsec))
		mintime = finish;
	    }
	}
      printf ("elapsed time: %jd.%09jd sec\n",
	      (intmax_t) mintime.tv_sec, (intmax_t) mintime.tv_nsec);
    }
#endif

  /* Return an error if the number of matches found is not match we
     expect.  */
  return cnt != expected;
}


static int
run_test_backwards (const char *expr, const char *mem, size_t memlen,
		    int icase, int expected)
{
#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
  struct timespec start;
  struct timespec finish;
#endif
  struct re_pattern_buffer re;
  const char *err;
  size_t offset;
  int cnt;

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
  if (use_clock && !timing)
    use_clock = clock_gettime (cl, &start) == 0;
#endif

  re_set_syntax ((RE_SYNTAX_POSIX_BASIC & ~RE_DOT_NEWLINE)
		 | RE_HAT_LISTS_NOT_NEWLINE
		 | (icase ? RE_ICASE : 0));

  memset (&re, 0, sizeof (re));
  re.fastmap = malloc (256);
  if (re.fastmap == NULL)
    error (EXIT_FAILURE, errno, "cannot allocate fastmap");

  err = re_compile_pattern (expr, strlen (expr), &re);
  if (err != NULL)
    error (EXIT_FAILURE, 0, "cannot compile expression: %s", err);

  if (re_compile_fastmap (&re))
    error (EXIT_FAILURE, 0, "couldn't compile fastmap");

  cnt = 0;
  offset = memlen;
  assert (mem[memlen] == '\0');
  while (offset <= memlen)
    {
      int start;
      const char *sp;
      const char *ep;

      start = re_search (&re, mem, memlen, offset, -offset, NULL);
      if (start == -1)
	break;

      if (start == -2)
	error (EXIT_FAILURE, 0, "internal error in re_search");

      sp = mem + start;
      while (sp > mem && sp[-1] != '\n')
	--sp;

      ep = mem + start;
      while (*ep != '\0' && *ep != '\n')
	++ep;

      printf ("match %d: \"%.*s\"\n", ++cnt, (int) (ep - sp), sp);

      offset = sp - 1 - mem;
    }

  regfree (&re);

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
  if (use_clock && !timing)
    {
      use_clock = clock_gettime (cl, &finish) == 0;
      if (use_clock)
	{
	  if (finish.tv_nsec < start.tv_nsec)
	    {
	      finish.tv_nsec -= start.tv_nsec - 1000000000;
	      finish.tv_sec -= 1 + start.tv_sec;
	    }
	  else
	    {
	      finish.tv_nsec -= start.tv_nsec;
	      finish.tv_sec -= start.tv_sec;
	    }

	  printf ("elapsed time: %jd.%09jd sec\n",
		  (intmax_t) finish.tv_sec, (intmax_t) finish.tv_nsec);
	}
    }

  if (use_clock && timing)
    {
      struct timespec mintime = { .tv_sec = 24 * 60 * 60 };

      for (int i = 0; i < 10; ++i)
	{
	  offset = memlen;
	  use_clock = clock_gettime (cl, &start) == 0;

	  if (!use_clock)
	    continue;

	  memset (&re, 0, sizeof (re));
	  re.fastmap = malloc (256);
	  if (re.fastmap == NULL)
	    continue;

	  err = re_compile_pattern (expr, strlen (expr), &re);
	  if (err != NULL)
	    continue;

	  if (re_compile_fastmap (&re))
	    {
	      regfree (&re);
	      continue;
	    }

	  while (offset <= memlen)
	    {
	      int start;
	      const char *sp;

	      start = re_search (&re, mem, memlen, offset, -offset, NULL);
	      if (start < -1)
		break;

	      sp = mem + start;
	      while (sp > mem && sp[-1] != '\n')
		--sp;

	      offset = sp - 1 - mem;
	    }

	  regfree (&re);

	  use_clock = clock_gettime (cl, &finish) == 0;
	  if (use_clock)
	    {
	      if (finish.tv_nsec < start.tv_nsec)
		{
		  finish.tv_nsec -= start.tv_nsec - 1000000000;
		  finish.tv_sec -= 1 + start.tv_sec;
		}
	      else
		{
		  finish.tv_nsec -= start.tv_nsec;
		  finish.tv_sec -= start.tv_sec;
		}
	      if (finish.tv_sec < mintime.tv_sec
		  || (finish.tv_sec == mintime.tv_sec
		      && finish.tv_nsec < mintime.tv_nsec))
		mintime = finish;
	    }
	}
      printf ("elapsed time: %jd.%09jd sec\n",
	      (intmax_t) mintime.tv_sec, (intmax_t) mintime.tv_nsec);
    }
#endif

  /* Return an error if the number of matches found is not match we
     expect.  */
  return cnt != expected;
}

/* If --timing is used we will need a larger timout.  */
#define TIMEOUT 50
#define CMDLINE_OPTIONS \
   {"timing", no_argument, &timing, 1 },
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
