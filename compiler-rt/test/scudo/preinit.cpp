// RUN: %clang_scudo %s -o %t
// RUN: %run %t 2>&1

// Verifies that calling malloc in a preinit_array function succeeds, and that
// the resulting pointer can be freed at program termination.

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void *global_p = nullptr;

void __init(void) {
  global_p = malloc(1);
  if (!global_p)
    exit(1);
}

void __fini(void) {
  if (global_p)
    free(global_p);
}

int main(int argc, char **argv)
{
  void *p = malloc(1);
  assert(p);
  free(p);

  return 0;
}

__attribute__((section(".preinit_array"), used))
  void (*__local_preinit)(void) = __init;
__attribute__((section(".fini_array"), used))
  void (*__local_fini)(void) = __fini;
