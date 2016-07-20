// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=212 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {   // CHECK: class Bar
public:
  Foo();      // CHECK: Bar();
};

Foo::Foo() {} // CHECK: Bar::Bar()

// Use grep -FUbo 'C' <file> to get the correct offset of foo when changing
// this file.
