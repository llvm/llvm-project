#include <stdio.h>

int g_var = 200;

int
main(int argc, char **argv)
{
  int test_var = 10;
  printf ("Set a breakpoint here: %d %d.\n", test_var, g_var);
  return 0;
}
