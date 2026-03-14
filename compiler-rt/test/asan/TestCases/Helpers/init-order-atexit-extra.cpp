#include <stdio.h>

class C {
 public:
  C() { value = 42; }
  ~C() { }
  int value;
};

C c;

void AccessC() {
  printf("C value: %d\n", c.value);
}

int main() { return 0; }
