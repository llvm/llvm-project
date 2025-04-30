#include <stdio.h>
#include <sys/types.h>


static int
check (FILE *fp, off_t o)
{
  int result = 0;
  if (feof (fp))
    {
      puts ("feof !");
      result = 1;
    }
  if (ferror (fp))
    {
      puts ("ferror !");
      result = 1;
    }
  if (ftello (fp) != o)
    {
      printf ("position = %lu, not %lu\n", (unsigned long int) ftello (fp),
	      (unsigned long int) o);
      result = 1;
    }
  return result;
}


static int
do_test (void)
{
  FILE *fp = tmpfile ();
  if (fp == NULL)
    {
      puts ("tmpfile failed");
      return 1;
    }
  if (check (fp, 0) != 0)
    return 1;

  puts ("going to write");
  if (fputs ("hello", fp) == EOF)
    {
      puts ("fputs failed");
      return 1;
    }
  if (check (fp, 5) != 0)
    return 1;

  puts ("going to rewind");
  rewind (fp);
  if (check (fp, 0) != 0)
    return 1;

  puts ("going to read char");
  int c = fgetc (fp);
  if (c != 'h')
    {
      printf ("read %c, not %c\n", c, 'h');
      return 1;
    }
  if (check (fp, 1) != 0)
    return 1;

  puts ("going to put back");
  if (ungetc (' ', fp) == EOF)
    {
      puts ("ungetc failed");
      return 1;
    }
  if (check (fp, 0) != 0)
    return 1;

  puts ("going to write again");
  if (fputs ("world", fp) == EOF)
    {
      puts ("2nd fputs failed");
      return 1;
    }
  if (check (fp, 5) != 0)
    return 1;

  puts ("going to rewind again");
  rewind (fp);
  if (check (fp, 0) != 0)
    return 1;

  if (fclose (fp) != 0)
    {
      puts ("fclose failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
