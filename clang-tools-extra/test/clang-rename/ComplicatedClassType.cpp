// Unsupported test.
// RUN: cat %s > %t.cpp
// FIXME: This test contains very simple constructions likely to be seen in any
// project and therefore passing this test is a slight sign of success.
// Currently, the test fails badly.

class Foo {                           // CHECK: class Bar {
 public:
  Foo(int value = 0) : x(value) {}    // Bar(int value=0) : x(value) {}

  Foo& operator++(int) {              // Bar& operator++(int) {
    x++;
    return *this;
  }

  bool operator<(Foo const& rhs) {    // bool operator<(Bar const &rhs) {
    return this->x < rhs.x;
  }

 private:
  int x;
};

int main() {
  Foo* Pointer = 0;                   // CHECK: Bar *Pointer = 0;
  Foo Variable = Foo(10);             // CHECK: Bar Variable = Bar(10);
  for (Foo it; it < Variable; it++) { // for (Bar it; it < Variable; it++) {}
  }
  return 0;
}
