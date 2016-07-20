// Currently unsupported test.
// RUN: cat %s > %t.cpp
// FIXME: clang-rename doesn't recognize symbol in class function definition.

class Foo {
public:
  void foo(int x);
};

void Foo::foo(int x) {}
//   ^ this one

int main() {
  Foo obj;
  obj.foo(0);
  return 0;
}
