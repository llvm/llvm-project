#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

class Base {
public:
  virtual int Foo() = 0;
};

class Derived1 : public Base {
public:
  int Foo() override { return 1; }
};

class Derived2 : public Base {
public:
  int Foo() override { return 2; }
};

class Derived3 : public Base {
public:
  int Foo() override { return 3; }
};

int main(int argc, char *argv[]) {
  long long sum = 0;
  int outerIters = atoi(argv[1]);
  int selector = atoi(argv[2]);

  Base *obj1 = new Derived1();
  Base *obj2 = new Derived2();
  Base *obj3 = new Derived3();

  for (int j = 0; j < outerIters; j++) {
    for (int i = 0; i < 10000; i++) {
      switch (selector) {
      case 1: sum += obj1->Foo();  break;
      case 2: sum += obj2->Foo();  break;
      case 3: sum += obj3->Foo();  break;
      }
    }
  }
  printf("%lld\n", sum);
}
