// RUN: clang-rename -offset=100 -new-name=Bar %s -- | FileCheck %s

class C {
public:
  static int Foo; // CHECK: static int Bar;
};

int foo(int x) { return 0; }
#define MACRO(a) foo(a)

int main() {
  C::Foo = 1;     // CHECK: C::Bar
  MACRO(C::Foo);    // CHECK: C::Bar
  int y = C::Foo; // CHECK: C::Bar
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
