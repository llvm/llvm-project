/* Test for ungetc bugs.  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#undef assert
#define assert(x) \
  if (!(x)) \
    { \
      fputs ("test failed: " #x "\n", stderr); \
      retval = 1; \
      goto the_end; \
    }

int
main (int argc, char *argv[])
{
  char name[] = "/tmp/tst-ungetc.XXXXXX";
  FILE *fp = NULL;
  int retval = 0;
  int c;
  char buffer[64];

  int fd = mkstemp (name);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      return 1;
    }
  close (fd);
  fp = fopen (name, "w");
  assert (fp != NULL)
  fputs ("bla", fp);
  fclose (fp);
  fp = NULL;

  fp = fopen (name, "r");
  assert (fp != NULL);
  assert (ungetc ('z', fp) == 'z');
  assert (getc (fp) == 'z');
  assert (getc (fp) == 'b');
  assert (getc (fp) == 'l');
  assert (ungetc ('m', fp) == 'm');
  assert (getc (fp) == 'm');
  assert ((c = getc (fp)) == 'a');
  assert (getc (fp) == EOF);
  assert (ungetc (c, fp) == c);
  assert (feof (fp) == 0);
  assert (getc (fp) == c);
  assert (getc (fp) == EOF);
  fclose (fp);
  fp = NULL;

  fp = fopen (name, "r");
  assert (fp != NULL);
  assert (getc (fp) == 'b');
  assert (getc (fp) == 'l');
  assert (ungetc ('b', fp) == 'b');
  assert (fread (buffer, 1, 64, fp) == 2);
  assert (buffer[0] == 'b');
  assert (buffer[1] == 'a');

the_end:
  if (fp != NULL)
    fclose (fp);
  unlink (name);

  return retval;
}
