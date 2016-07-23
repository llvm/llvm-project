// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=136 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {             // CHECK: class Bar {
public:
  void foo(int x);
};

void Foo::foo(int x) {} // CHECK: void Bar::foo(int x) {}
