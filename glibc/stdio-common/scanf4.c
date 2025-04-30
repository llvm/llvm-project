#include <stdio.h>
#include <stdlib.h>
#include <libc-diag.h>

int
main(int arc, char *argv[])
{
  int res;
  unsigned int val;

  FILE *fp = fopen ("/dev/null", "r");

  val = 0;
  res = fscanf(fp, "%n", &val);

  printf("Result of fscanf %%n = %d\n", res);
  printf("Scanned format = %d\n", val);

  /* We're testing exactly the case the warning is for.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat-zero-length");

  res = fscanf(fp, "");

  DIAG_POP_NEEDS_COMMENT;

  printf("Result of fscanf \"\" = %d\n", res);
  if (res != 0)
    abort ();

  res = fscanf(fp, "BLURB");
  printf("Result of fscanf \"BLURB\" = %d\n", res);
  if (res >= 0)
    abort ();

  fclose (fp);

  return 0;
  return 0;
}
