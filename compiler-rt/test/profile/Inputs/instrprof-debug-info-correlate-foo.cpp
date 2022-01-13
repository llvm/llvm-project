#include "instrprof-debug-info-correlate-bar.h"

int foo(int a) {
  if (a % 2)
    return 4 * a + 1;
  return bar(a);
}
