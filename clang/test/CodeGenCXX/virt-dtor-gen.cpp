// RUN: %clang_cc1 -o - -triple %itanium_abi_triple -emit-llvm %s | FileCheck %s
// PR5483

// Make sure we generate all three forms of the destructor when it is virtual.
class Foo {
  virtual ~Foo();
};
Foo::~Foo() {}

// CHECK-LABEL: define {{.*}}void @_ZN3FooD0Ev(%class.Foo* {{[^,]*}} %this) unnamed_addr
