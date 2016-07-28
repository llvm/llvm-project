// RUN: clang-rename -offset=143 -new-name=Bar %s -- | FileCheck %s

class Foo {};             // CHECK: class Bar {};

class Baz {
  operator Foo() const {  // CHECK: operator Bar() const {
// offset  ^
    Foo foo;              // CHECK: Bar foo;
    return foo;
  }
};
