// RUN: clang-rename -offset=102 -new-name=Bar %s -- | FileCheck %s

class Baz {};

class Qux {
  Baz Foo;            // CHECK: Baz Bar;
public:
  Qux();
};

Qux::Qux() : Foo() {} // CHECK: Qux::Qux() : Bar() {}

// Use grep -FUbo 'Foo' <file> to get the correct offset of foo when changing
// this file.
