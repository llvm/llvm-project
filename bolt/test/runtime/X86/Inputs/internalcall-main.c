// BOLT test case

#include <stdio.h>

typedef unsigned long long (*FP)();

extern FP getCallback();
extern FP getCallback2();
extern FP getCallback3();

int main() {
  printf("Case 1: Result is: %llX\n", (*getCallback())());
  printf("Case 2: Result is: %llX\n", (*getCallback2())());
  printf("Case 3: Result is: %llX\n", (*getCallback3())());
  return 0;
}
