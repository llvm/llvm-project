// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=205 -new-name=Bar %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {};             // CHECK: class Bar {};

class Baz {
  operator Foo() const {  // CHECK: operator Bar() const {
// offset  ^
    Foo foo;              // CHECK: Bar foo;
    return foo;
  }
};
