// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=163 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Baz {};

class Qux {
  Baz Foo;            // CHECK: Baz Bar;
public:
  Qux();
};

Qux::Qux() : Foo() {} // CHECK: Qux::Qux() : Bar() {}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
