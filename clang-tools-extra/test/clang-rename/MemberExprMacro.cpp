// RUN: clang-rename -offset=95 -new-name=Bar %s -- | FileCheck %s

class Baz {
public:
  int Foo;          // CHECK: int Bar;
};

int qux(int x) { return 0; }
#define MACRO(a) qux(a)

int main() {
  Baz baz;
  baz.Foo = 1;      // CHECK: baz.Bar = 1;
  MACRO(baz.Foo);   // CHECK: MACRO(baz.Bar);
  int y = baz.Foo;  // CHECK: int y = baz.Bar;
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
