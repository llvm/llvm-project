// Currently unsupported test.
// RUN: cat %s > %t.cpp
// FIXME: clang-rename should handle conversions from a class type to another
// type.

class Foo {};             // CHECK: class Bar {};

class Baz {               // CHECK: class Bar {
  operator Foo() const {  // CHECK: operator Bar() const {
    Foo foo;              // CHECK: Bar foo;
    return foo;
  }
};
