#include <stdio.h>
#include <stdlib.h>

int
main(int arc, char *argv[])
{
  int n, res;
  unsigned int val;
  char s[] = "111";

  val = n = -1;
  res = sscanf(s, "%u %n", &val, &n);
  printf("Result of sscanf = %d\n", res);
  printf("Scanned format %%u = %u\n", val);
  printf("Possibly scanned format %%n = %d\n", n);
  if (n != 3 || val != 111 || res != 1)
    abort ();

  val = n = -1;
  res = sscanf(s, "%u%n", &val, &n);
  printf("Result of sscanf = %d\n", res);
  printf("Scanned format %%u = %u\n", val);
  printf("Possibly scanned format %%n = %d\n", n);
  if (n != 3 || val != 111 || res != 1)
    abort ();

  return 0;
}
