#include <stdio.h>

int bar(int x, int y) {
  if (x % 3) {
    return x - y;
  }
  return x + y;
}

void foo() {
  int s, i = 0;
  while (i++ < 4000 * 4000)
    if (i % 91) s = bar(i, s); else s += 30;
  printf("sum is %d\n", s);
}

int main() {
  foo();
  return 0;
}
