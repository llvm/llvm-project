// RUN: clang-rename -offset=74 -new-name=Bar %s -- | FileCheck %s

class Foo {             // CHECK: class Bar {
public:
  void foo(int x);
};

void Foo::foo(int x) {} // CHECK: void Bar::foo(int x) {}
