#include <stdio.h>
#include <stdlib.h>

int total = 0;

int inc(int x) {
  switch (x) {
    case 0: total += 1 + 0; return 1;
    case 1: total += 1 + 1; return 2;
    case 2: total += 1 + 2; return 3;
    case 3: total += 1 + 3; return 4;
    case 4: total += 1 + 4; return 5;
    case 5: total += 1 + 5; return 6;
    default: return x + 1;
  }
}

int inc_dup(int x) {
  switch (x) {
    case 0: total += 2 + 0; return 1;
    case 1: total += 2 + 1; return 2;
    case 2: total += 2 + 2; return 3;
    case 3: total += 2 + 3; return 4;
    case 4: total += 2 + 4; return 5;
    case 5: total += 2 + 5; return 6;
    default: return x + 1;
  }
}

int main() {
  int c = 0;
  for (int i = 0; i < 10000000; ++i) {
    int a = rand() % 7;
    int b = rand() % 7;
    c += inc(a) - 2*inc_dup(b);
  }
  return c == 0;
}
