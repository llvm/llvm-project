// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=220 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

// Forward declaration.
class Foo;                            // CHECK: class Bar;

class Foo {                           // CHECK: class Bar {
public:
  Foo(int value = 0) : x(value) {}    // CHECK: Bar(int value = 0) : x(value) {}

  Foo &operator++(int) {              // CHECK: Bar &operator++(int) {
    x++;
    return *this;
  }

  bool operator<(Foo const &rhs) {    // CHECK: bool operator<(Bar const &rhs) {
    return this->x < rhs.x;
  }

private:
  int x;
};

int main() {
  Foo *Pointer = 0;                   // CHECK: Bar *Pointer = 0;
  Foo Variable = Foo(10);             // CHECK: Bar Variable = Bar(10);
  for (Foo it; it < Variable; it++) { // CHECK: for (Bar it; it < Variable; it++) {
  }
  return 0;
}
