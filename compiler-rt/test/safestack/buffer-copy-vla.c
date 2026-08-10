// RUN: %clang_safestack %s -o %t
// RUN: %run %t

#include "utils.h"

// Test that loads/stores work correctly for VLAs on the unsafe stack.

int main(int argc, char **argv)
{
  int i = 128;
  break_optimization(&i);
  char buffer[i];

  // check that we can write to a buffer
  for (i = 0; argv[0][i] && i < sizeof (buffer) - 1; ++i)
    buffer[i] = argv[0][i];
  buffer[i] = '\0';

  break_optimization(buffer);

  // check that we can read from a buffer
  for (i = 0; argv[0][i] && i < sizeof (buffer) - 1; ++i)
    if (buffer[i] != argv[0][i])
      return 1;
  return 0;
}
