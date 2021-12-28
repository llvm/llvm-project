#include "instrprof-debug-info-correlate-bar.h"

typedef int (*FP)(int);
FP Fps[2] = {foo, bar};

int main() {
  for (int i = 0; i < 5; i++)
    Fps[i % 2](i);
  return 0;
}
