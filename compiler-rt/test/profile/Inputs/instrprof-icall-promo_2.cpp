#include "instrprof-icall-promo.h"
extern int ref(A *);

int A::bar() { return 2; }

extern A *ap;
int test() {
  for (int i = 0; i < 10000; i++) ap->foo();
  return ref(ap);
}

int main() {
  test();
  return 0;
}
