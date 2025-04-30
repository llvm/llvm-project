/* If stdio is working correctly, after this is run infile and outfile
   will have the same contents.  If the bug (found in GNU C library 0.3)
   exhibits itself, outfile will be missing the 2nd through 1023rd
   characters.  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static char buf[8192];

int
main (void)
{
  FILE *in;
  FILE *out;
  static char inname[] = OBJPFX "bug5test.in";
  static char outname[] = OBJPFX "bug5test.out";
  char *printbuf;
  size_t i;
  int result;

  /* Create a test file.  */
  in = fopen (inname, "w+");
  if (in == NULL)
    {
      perror (inname);
      return 1;
    }
  for (i = 0; i < 1000; ++i)
    fprintf (in, "%Zu\n", i);

  out = fopen (outname, "w");
  if (out == NULL)
    {
      perror (outname);
      return 1;
    }
  if (fseek (in, 0L, SEEK_SET) != 0)
    abort ();
  putc (getc (in), out);
  i = fread (buf, 1, sizeof (buf), in);
  if (i == 0)
    {
      perror ("fread");
      return 1;
    }
  if (fwrite (buf, 1, i, out) != i)
    {
      perror ("fwrite");
      return 1;
    }
  fclose (in);
  fclose (out);

  puts ("There should be no further output from this test.");
  fflush (stdout);

  /* We must remove this entry to assure the `cmp' binary does not use
     the perhaps incompatible new shared libraries.  */
  unsetenv ("LD_LIBRARY_PATH");

  asprintf (&printbuf, "cmp %s %s", inname, outname);
  result = system (printbuf);
  remove (inname);
  remove (outname);

  exit ((result != 0));
}
