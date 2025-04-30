#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int
do_test (void)
{
  int fd = dup (fileno (stdout));
  if (fd <= 1)
    {
      puts ("dup failed");
      return 1;
    }

  FILE *f1 = fdopen (fd, "w");
  if (f1 == NULL)
    {
      printf ("fdopen failed: %m\n");
      return 1;
    }

  fclose (stdout);

  FILE *f2 = popen ("echo test1", "r");
  if (f2 == NULL)
    {
      fprintf (f1, "1st popen failed: %m\n");
      return 1;
    }
  FILE *f3 = popen ("echo test2", "r");
  if (f2 == NULL || f3 == NULL)
    {
      fprintf (f1, "2nd popen failed: %m\n");
      return 1;
    }

  char *line = NULL;
  size_t len = 0;
  int result = 0;
  if (getline (&line, &len, f2) != 6)
    {
      fputs ("could not read line from 1st popen\n", f1);
      result = 1;
    }
  else if (strcmp (line, "test1\n") != 0)
    {
      fprintf (f1, "read \"%s\"\n", line);
      result = 1;
    }

  if (getline (&line, &len, f2) != -1)
    {
      fputs ("second getline did not return -1\n", f1);
      result = 1;
    }

  if (getline (&line, &len, f3) != 6)
    {
      fputs ("could not read line from 2nd popen\n", f1);
      result = 1;
    }
  else if (strcmp (line, "test2\n") != 0)
    {
      fprintf (f1, "read \"%s\"\n", line);
      result = 1;
    }

  if (getline (&line, &len, f3) != -1)
    {
      fputs ("second getline did not return -1\n", f1);
      result = 1;
    }

  int ret = pclose (f2);
  if (ret != 0)
    {
      fprintf (f1, "1st pclose returned %d\n", ret);
      result = 1;
    }

  ret = pclose (f3);
  if (ret != 0)
    {
      fprintf (f1, "2nd pclose returned %d\n", ret);
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
