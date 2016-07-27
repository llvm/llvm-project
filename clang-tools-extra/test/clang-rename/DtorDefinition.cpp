// RUN: clang-rename -offset=158 -new-name=Bar %s -- | FileCheck %s

class Foo {     // CHECK: class Bar {
public:
  ~Foo();       // CHECK: ~Bar();
};

Foo::~Foo() {}  // CHECK: Bar::~Bar()


// Use grep -FUbo 'Foo' <file> to get the correct offset of Foo when changing
// this file.
