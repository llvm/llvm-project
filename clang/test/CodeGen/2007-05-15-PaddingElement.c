// PR 1419

// RUN: %clang_cc1   -O2 %s -emit-llvm -o - | grep "ret i32 1"
struct A {
  short x;
  long long :0;
};

struct B {
  char a;
  char b;
  unsigned char i;
};

union X { struct A a; struct B b; };

int check(void) {
  union X x, y;

  y.b.i = 0xff;
  x = y;
  return (x.b.i == 0xff);
}
