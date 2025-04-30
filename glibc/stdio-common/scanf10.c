#include <stdio.h>
#include <string.h>

int
main (int argc, char *argv[])
{
  const char teststring[] = "<tag `word'>";
  int retc, a, b;

  retc = sscanf (teststring, "<%*s `%n%*s%n'>", &a, &b);

  printf ("retc=%d a=%d b=%d\n", retc, a, b);

  return retc == -1 && a == 6 && b == 12 ? 0 : 1;
}
