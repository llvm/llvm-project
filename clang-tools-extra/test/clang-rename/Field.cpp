// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=148 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Baz {
  int Foo;              // CHECK: Bar;
public:
  Baz();
};

Baz::Baz() : Foo(0) {}  // CHECK: Baz::Baz() : Bar(0) {}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
