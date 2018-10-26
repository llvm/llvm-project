// Check that mallopt does not return invalid values (ex. -1).
// RUN: %clangxx -O2 %s -o %t && %run %t
#include <assert.h>
#include <malloc.h>

int main() {
  int res = mallopt(M_ARENA_MAX, 0);
  assert(res == 0 || res == 1);
}
