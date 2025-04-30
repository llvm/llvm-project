#include <fcntl.h>
#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
static clockid_t cl;
static int use_clock;
#endif

static int
do_test (void)
{
#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
# if _POSIX_CPUTIME == 0
  if (sysconf (_SC_CPUTIME) < 0)
    use_clock = 0;
  else
# endif
    /* See whether we can use the CPU clock.  */
    use_clock = clock_getcpuclockid (0, &cl) == 0;
#endif

  static const char *pat[] = {
    ".?.?.?.?.?.?.?Log\\.13",
    "(.?)(.?)(.?)(.?)(.?)(.?)(.?)Log\\.13",
    "((((((((((.?))))))))))((((((((((.?))))))))))((((((((((.?))))))))))"
    "((((((((((.?))))))))))((((((((((.?))))))))))((((((((((.?))))))))))"
    "((((((((((.?))))))))))Log\\.13" };

  int fd = open ("../ChangeLog.old/ChangeLog.14", O_RDONLY);
  if (fd < 0)
    {
      printf ("Couldn't open ChangeLog.14: %m\n");
      return 1;
    }

  struct stat64 st;
  if (fstat64 (fd, &st) < 0)
    {
      printf ("Couldn't fstat ChangeLog.14: %m\n");
      return 1;
    }

  char *buf = malloc (st.st_size + 1);
  if (buf == NULL)
    {
      printf ("Couldn't allocate buffer: %m\n");
      return 1;
    }

  if (read (fd, buf, st.st_size) != (ssize_t) st.st_size)
    {
      puts ("Couldn't read ChangeLog.14");
      return 1;
    }

  close (fd);
  buf[st.st_size] = '\0';

  setlocale (LC_ALL, "de_DE.UTF-8");

  char *string = buf;
  size_t len = st.st_size;

#ifndef WHOLE_FILE_TIMING
  /* Don't search the whole file normally, it takes too long.  */
  if (len > 500000 + 64)
    {
      string += 500000;
      len -= 500000;
    }
#endif

  for (int testno = 0; testno < 4; ++testno)
    for (int i = 0; i < sizeof (pat) / sizeof (pat[0]); ++i)
      {
	printf ("test %d pattern %d", testno, i);

	regex_t rbuf;
	struct re_pattern_buffer rpbuf;
	int err;
	if (testno < 2)
	  {
	    err = regcomp (&rbuf, pat[i],
			   REG_EXTENDED | (testno ? REG_NOSUB : 0));
	    if (err != 0)
	      {
		putchar ('\n');
		char errstr[300];
		regerror (err, &rbuf, errstr, sizeof (errstr));
		puts (errstr);
		return err;
	      }
	  }
	else
	  {
	    re_set_syntax (RE_SYNTAX_POSIX_EGREP
			   | (testno == 3 ? RE_NO_SUB : 0));

	    memset (&rpbuf, 0, sizeof (rpbuf));
	    const char *s = re_compile_pattern (pat[i], strlen (pat[i]),
						&rpbuf);
	    if (s != NULL)
	      {
		printf ("\n%s\n", s);
		return 1;
	      }

	    /* Just so that this can be tested with earlier glibc as well.  */
	    if (testno == 3)
	      rpbuf.no_sub = 1;
	  }

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
      struct timespec start, stop;
      if (use_clock)
	use_clock = clock_gettime (cl, &start) == 0;
#endif

      if (testno < 2)
	{
	  regmatch_t pmatch[71];
	  err = regexec (&rbuf, string, 71, pmatch, 0);
	  if (err == REG_NOMATCH)
	    {
	      puts ("\nregexec failed");
	      return 1;
	    }

	  if (testno == 0)
	    {
	      if (pmatch[0].rm_eo != pmatch[0].rm_so + 13
		  || pmatch[0].rm_eo > len
		  || pmatch[0].rm_so < len - 100
		  || strncmp (string + pmatch[0].rm_so,
			      " ChangeLog.13 for earlier changes",
			      sizeof " ChangeLog.13 for earlier changes" - 1)
		     != 0)
		{
		  puts ("\nregexec without REG_NOSUB did not find the correct match");
		  return 1;
		}

	      if (i > 0)
		for (int j = 0, l = 1; j < 7; ++j)
		  for (int k = 0; k < (i == 1 ? 1 : 10); ++k, ++l)
		    if (pmatch[l].rm_so != pmatch[0].rm_so + j
			|| pmatch[l].rm_eo != pmatch[l].rm_so + 1)
		      {
			printf ("\npmatch[%d] incorrect\n", l);
			return 1;
		      }
	    }
	}
      else
	{
	  struct re_registers regs;

	  memset (&regs, 0, sizeof (regs));
	  int match = re_search (&rpbuf, string, len, 0, len,
				 &regs);
	  if (match < 0)
	    {
	      puts ("\nre_search failed");
	      return 1;
	    }

	  if (match + 13 > len
	      || match < len - 100
	      || strncmp (string + match,
			  " ChangeLog.13 for earlier changes",
			  sizeof " ChangeLog.13 for earlier changes" - 1)
		  != 0)
	    {
	      puts ("\nre_search did not find the correct match");
	      return 1;
	    }

	  if (testno == 2)
	    {
	      if (regs.num_regs != 2 + (i == 0 ? 0 : i == 1 ? 7 : 70))
		{
		  printf ("\nincorrect num_regs %d\n", regs.num_regs);
		  return 1;
		}

	      if (regs.start[0] != match || regs.end[0] != match + 13)
		{
		  printf ("\nincorrect regs.{start,end}[0] = { %d, %d}\n",
			  regs.start[0], regs.end[0]);
		  return 1;
		}

	      if (regs.start[regs.num_regs - 1] != -1
		  || regs.end[regs.num_regs - 1] != -1)
		{
		  puts ("\nincorrect regs.{start,end}[num_regs - 1]");
		  return 1;
		}

	      if (i > 0)
		for (int j = 0, l = 1; j < 7; ++j)
		  for (int k = 0; k < (i == 1 ? 1 : 10); ++k, ++l)
		    if (regs.start[l] != match + j
			|| regs.end[l] != regs.start[l] + 1)
		      {
			printf ("\nregs.{start,end}[%d] incorrect\n", l);
			return 1;
		      }
	    }
	}

#if defined _POSIX_CPUTIME && _POSIX_CPUTIME >= 0
      if (use_clock)
	use_clock = clock_gettime (cl, &stop) == 0;
      if (use_clock)
	{
	  stop.tv_sec -= start.tv_sec;
	  if (stop.tv_nsec < start.tv_nsec)
	    {
	      stop.tv_sec--;
	      stop.tv_nsec += 1000000000 - start.tv_nsec;
	    }
	  else
	    stop.tv_nsec -= start.tv_nsec;
	  printf (": %ld.%09lds\n", (long) stop.tv_sec, (long) stop.tv_nsec);
	}
      else
#endif
	putchar ('\n');

      if (testno < 2)
	regfree (&rbuf);
      else
	regfree (&rpbuf);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
