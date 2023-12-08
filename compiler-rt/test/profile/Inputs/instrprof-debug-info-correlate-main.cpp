#include "instrprof-debug-info-correlate-bar.h"

typedef int (*FP)(int);
FP Fps[3] = {foo, bar, unused};

int main() {
  for (int i = 0; i < 5; i++)
    Fps[i % 2](i);
  return 0;
}
