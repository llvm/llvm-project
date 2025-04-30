/* Test for fdopen bugs.  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#undef assert
#define assert(x) \
  if (!(x)) \
    { \
      fputs ("test failed: " #x "\n", stderr); \
      retval = 1; \
      goto the_end; \
    }

char buffer[256];

int
main (int argc, char *argv[])
{
  char name[] = "/tmp/tst-fdopen.XXXXXX";
  FILE *fp = NULL;
  int retval = 0;
  int fd;

  fd = mkstemp (name);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      return 1;
    }
  close (fd);
  fp = fopen (name, "w");
  assert (fp != NULL)
  fputs ("foobar and baz", fp);
  fclose (fp);
  fp = NULL;

  fd = open (name, O_RDONLY);
  assert (fd != -1);
  assert (lseek (fd, 5, SEEK_SET) == 5);
  /* The file position indicator associated with the new stream is set to
     the position indicated by the file offset associated with the file
     descriptor.  */
  fp = fdopen (fd, "r");
  assert (fp != NULL);
  assert (getc (fp) == 'r');
  assert (getc (fp) == ' ');

the_end:
  if (fp != NULL)
    fclose (fp);
  unlink (name);

  return retval;
}
