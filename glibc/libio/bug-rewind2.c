#include <stdio.h>
#include <wchar.h>


static int fd;

static void prepare (void);
#define PREPARE(argc, argv) prepare ()


#define TEST_FUNCTION do_test ()
static int do_test (void);
#include "../test-skeleton.c"


static void
prepare (void)
{
  fd = create_temp_file ("wrewind2.", NULL);
  if (fd == -1)
    exit (3);
}


static int
do_test (void)
{
  wchar_t dummy[10];
  int ret = 0;
  FILE *fp;
  int result = 0;

  fp = fdopen (fd, "w+");
  if (fp == NULL)
    {
      puts ("fopen(""testfile"", ""w+"") returned NULL.");
      return 1;
    }
  else
    {
      fwprintf (fp, L"abcd");
      printf ("current pos = %ld\n", ftell (fp));
      if (ftell (fp) != 4)
	result = 1;

      rewind (fp);
      ret = fwscanf (fp, L"%c", dummy);
      if (ret != 1)
	{
	  printf ("fwscanf returned %d, expected 1\n", ret);
	  result = 1;
	}

      printf ("current pos = %ld\n", ftell (fp));
      if (ftell (fp) != 1)
	result = 1;

      rewind (fp);
      printf ("current pos = %ld\n", ftell (fp));
      if (ftell (fp) != 0)
	result = 1;

      fclose (fp);
    }

  return result;
}
