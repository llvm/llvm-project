// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s

class A { int a; };
class B {
public:
  B() { a = new A; }
  ~B() { delete a; }
private:
  A *a;
};

void fn(B b);

int i;
void foo() {
  if (i) {
    B b1;
    fn (b1);
  }
}
// Check there is a line number entry for line 19 where b1 is destructed.
// CHECK: !DILocation(line: 19,
