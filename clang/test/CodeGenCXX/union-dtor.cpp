// RUN: %clang_cc1 -std=c++11 %s -S -o - -emit-llvm | FileCheck %s

// PR10304: destructors should not call destructors for variant members.

template<bool b = false>
struct Foo {
  Foo() { static_assert(b, "Foo::Foo used"); }
  ~Foo() { static_assert(b, "Foo::~Foo used"); }
};

struct Bar {
  Bar();
  ~Bar();
};

union FooBar {
  FooBar() {}
  ~FooBar() {}
  Foo<> foo;
  Bar bar;
};

struct Variant {
  Variant() {}
  ~Variant() {}
  union {
    Foo<> foo;
    Bar bar;
  };
};

FooBar foobar;
Variant variant;

// The ctor and dtor of Foo<> and Bar should not be mentioned in the resulting
// code.
//
// CHECK-NOT: 3FooILb1EEC1
// CHECK-NOT: 3BarC1
//
// CHECK-NOT: 3FooILb1EED1
// CHECK-NOT: 3BarD1
