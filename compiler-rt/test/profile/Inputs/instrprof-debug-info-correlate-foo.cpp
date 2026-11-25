#include "instrprof-debug-info-correlate-bar.h"

// Has no debug info, but its counters must still be correlated.
__attribute__((nodebug)) static int baz(int a) { return a + 1; }

int foo(int a) {
  if (a % 2)
    return 4 * a + 1;
  return bar(baz(a));
}

int unused(int a) { return a * a; }
