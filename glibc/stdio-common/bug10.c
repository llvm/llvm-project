#include <stdio.h>

int
main (int arc, char *argv[])
{
  int n, res;
  unsigned int val;
  char s[] = "111";
  int result = 0;

  n = 0;
  res = sscanf(s, "%u %n", &val, &n);

  printf("Result of sscanf = %d\n", res);
  printf("Scanned format %%u = %u\n", val);
  printf("Possibly scanned format %%n = %d\n", n);
  result |= res != 1 || val != 111 || n != 3;


  result |= sscanf ("", " %n", &n) == EOF;

  puts (result ? "Test failed" : "All tests passed");

  return result;
}
