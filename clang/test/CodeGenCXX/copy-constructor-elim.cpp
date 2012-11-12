// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep "_ZN1CC1ERK1C" %t | count 0
// RUN: grep "_ZN1SC1ERK1S" %t | count 0

extern "C" int printf(...);


struct C {
  C() : iC(6) {printf("C()\n"); }
  C(const C& c) { printf("C(const C& c)\n"); }
  int iC;
};

C foo() {
  return C();
};

class X { // ...
public: 
  X(int) {}
  X(const X&, int i = 1, int j = 2, C c = foo()) {
    printf("X(const X&, %d, %d, %d)\n", i, j, c.iC);
  }
};


struct S {
  S();
};

S::S() { printf("S()\n"); }

void Call(S) {};

int main() {
  X a(1);
  X b(a, 2);
  X c = b;
  X d(a, 5, 6);
  S s;
  Call(s);
}
