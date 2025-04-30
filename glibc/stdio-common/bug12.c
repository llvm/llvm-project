#include <stdio.h>
#include <string.h>

char x[4096], z[4096], b[21], m[4096 * 4];

int
main (void)
{
  FILE *f = tmpfile ();
  int i, failed = 0;

  memset (x, 'x', 4096);
  memset (z, 'z', 4096);
  b[20] = 0;

  for (i = 0; i <= 5; i++)
    {
      fwrite (x, 4096, 1, f);
      fwrite (z, 4096, 1, f);
    }
  rewind (f);

  fread (m, 4096 * 4 - 10, 1, f);
  fread (b, 20, 1, f);
  printf ("got %s (should be %s)\n", b, "zzzzzzzzzzxxxxxxxxxx");
  if (strcmp (b, "zzzzzzzzzzxxxxxxxxxx"))
    failed = 1;

  fseek (f, -40, SEEK_CUR);
  fread (b, 20, 1, f);
  printf ("got %s (should be %s)\n", b, "zzzzzzzzzzzzzzzzzzzz");
  if (strcmp (b, "zzzzzzzzzzzzzzzzzzzz"))
    failed = 1;

  fread (b, 20, 1, f);
  printf ("got %s (should be %s)\n", b, "zzzzzzzzzzxxxxxxxxxx");
  if (strcmp (b, "zzzzzzzzzzxxxxxxxxxx"))
    failed = 1;

  fread (b, 20, 1, f);
  printf ("got %s (should be %s)\n", b, "xxxxxxxxxxxxxxxxxxxx");
  if (strcmp (b, "xxxxxxxxxxxxxxxxxxxx"))
    failed = 1;

  return failed;
}
