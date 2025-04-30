#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
main (int argc, char *argv[])
{
  int ret;
  char buf [1024] = "Ooops";

  ret = sscanf ("static char Term_bits[] = {", "static char %s = {", buf);
  printf ("ret: %d, name: %s\n", ret, buf);
  if (ret != 1 || strcmp (buf, "Term_bits[]") != 0)
    abort ();
  return 0;
}
