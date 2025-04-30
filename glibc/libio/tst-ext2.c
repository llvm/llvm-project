#include <stdio.h>
#include <stdio_ext.h>


static char *fname;

#define PREPARE(argc, argv) \
  do {									\
    int fd = create_temp_file ("tst-ext2", &fname);			\
    if (fd == -1)							\
      {									\
	puts ("cannot create temporary file");				\
	exit (1);							\
      }									\
    close (fd);								\
  } while (0)


static int
do_test (void)
{
  int res = 0;

  FILE *fp;

  fp = fopen (fname, "w");
  printf ("Initial state for write-only stream: %d %d\n",
          __freading (fp) != 0, __fwriting (fp) != 0);
  res |= ((__freading (fp) != 0) != 0
	  || (__fwriting (fp) != 0) != 1);
  fclose (fp);

  fp = fopen (fname, "r");
  printf ("Initial state for read-only stream:  %d %d\n",
          __freading (fp) != 0, __fwriting (fp) != 0);
  res |= ((__freading (fp) != 0) != 1
	  || (__fwriting (fp) != 0) != 0);
  fclose (fp);

  fp = fopen (fname, "r+");
  printf ("Initial state for read-write stream: %d %d\n",
          __freading (fp) != 0, __fwriting (fp) != 0);
  res |= ((__freading (fp) != 0) != 0
	  || (__fwriting (fp) != 0) != 0);
  fclose (fp);

  fp = fopen (fname, "w+");
  printf ("Initial state for read-write stream: %d %d\n",
          __freading (fp) != 0, __fwriting (fp) != 0);
  res |= ((__freading (fp) != 0) != 0
	  || (__fwriting (fp) != 0) != 0);
  fclose (fp);

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
