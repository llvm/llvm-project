#include <stdio.h>
#include <stdlib.h>
#include <libc-diag.h>

int
main (int argc, char *argv[])
{
  long long int n;
  int ret;

  n = -1;
  ret = sscanf ("1000", "%lld", &n);
  printf ("%%lld: ret: %d, n: %Ld\n", ret, n);
  if (ret != 1 || n != 1000L)
    abort ();

  n = -2;

  /* We are testing a corner case of the scanf format string here.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat");
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat-extra-args");

  ret = sscanf ("1000", "%llld", &n);

  DIAG_POP_NEEDS_COMMENT;

  printf ("%%llld: ret: %d, n: %Ld\n", ret, n);
  if (ret > 0 || n >= 0L)
    abort ();

  return 0;
}
