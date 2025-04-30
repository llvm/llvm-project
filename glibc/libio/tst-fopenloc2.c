#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wchar.h>


static const struct
{
  const char *enc;
  const char *data;
  size_t datalen;
  const wchar_t *expected;
  size_t expectedlen;
} tests[] =
  {
    { "UCS-4LE", "a\0\0\0b\0\0\0", 8, L"ab", 2 },
    { "UCS-4BE", "\0\0\0a\0\0\0b", 8, L"ab", 2 },
  };
#define ntests (sizeof (tests) / sizeof (tests[0]))


static int do_test (void);
#define TEST_FUNCTION do_test ()

static void prepare (void);
#define PREPARE(argc, argv) prepare ();

#include "../test-skeleton.c"


static int fd;
static char *tmpname;


static void
prepare (void)
{
  fd = create_temp_file ("tst-fopenloc2", &tmpname);
  if (fd == -1)
    {
      puts ("cannot open temp file");
      exit (1);
    }
}


static int
do_test (void)
{
  for (int i = 0; i < ntests; ++i)
    {
      if (ftruncate (fd, 0) != 0)
	{
	  printf ("ftruncate in round %d failed\n", i + 1);
	  return 1;
	}

      if (TEMP_FAILURE_RETRY (write (fd, tests[i].data, tests[i].datalen))
	  != tests[i].datalen)
	{
	  printf ("write in round %d failed\n", i + 1);
	  return 1;
	}

      if (lseek (fd, 0, SEEK_SET) != 0)
	{
	  printf ("lseek in round %d failed\n", i + 1);
	  return 1;
	}

      char *ccs;
      if (asprintf (&ccs, "r,ccs=%s", tests[i].enc) == -1)
	{
	  printf ("asprintf in round %d failed\n", i + 1);
	  return 1;
	}

      FILE *fp = fopen (tmpname, ccs);
      if (fp == NULL)
	{
	  printf ("fopen in round %d failed\n", i + 1);
	  return 1;
	}

#define LINELEN 100
      wchar_t line[LINELEN];
      if (fgetws (line, LINELEN, fp) != line)
	{
	  printf ("fgetws in round %d failed\n", i + 1);
	  return 1;
	}

      if (wcslen (line) != tests[i].expectedlen)
	{
	  printf ("round %d: expected length %zu, got length %zu\n",
		  i + 1, tests[i].expectedlen, wcslen (line));
	  return 1;
	}

      if (wcscmp (tests[i].expected, line) != 0)
	{
	  printf ("round %d: expected L\"%ls\", got L\"%ls\"\n",
		  i + 1, tests[i].expected, line);
	  return 1;
	}

      fclose (fp);

      free (ccs);
    }

  close (fd);

  return 0;
}
