// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=135 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Foo {
public:
  Foo();
  ~Foo(); // CHECK: ~Bar();
};

Foo::Foo() {
}

Foo::~Foo() { // CHECK: Bar::~Bar()
}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
