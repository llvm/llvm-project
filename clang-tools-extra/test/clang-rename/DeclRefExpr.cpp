// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=161 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

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

// Use grep -FUbo 'X' <file> to get the correct offset of foo when changing
// this file.
