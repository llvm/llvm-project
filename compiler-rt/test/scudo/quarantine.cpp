// RUN: %clang_scudo %s -o %t
// RUN: SCUDO_OPTIONS=QuarantineSizeMb=1 %run %t 2>&1

// Tests that the quarantine prevents a chunk from being reused right away.
// Also tests that a chunk will eventually become available again for
// allocation when the recycling criteria has been met.

#include <malloc.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  void *p, *old_p;
  size_t size = 1U << 16;

  // The delayed freelist will prevent a chunk from being available right away
  p = malloc(size);
  if (!p)
    return 1;
  old_p = p;
  free(p);
  p = malloc(size);
  if (!p)
    return 1;
  if (old_p == p)
    return 1;
  free(p);

  // Eventually the chunk should become available again
  bool found = false;
  for (int i = 0; i < 0x100 && found == false; i++) {
    p = malloc(size);
    if (!p)
      return 1;
    found = (p == old_p);
    free(p);
  }
  if (found == false)
    return 1;

  return 0;
}
