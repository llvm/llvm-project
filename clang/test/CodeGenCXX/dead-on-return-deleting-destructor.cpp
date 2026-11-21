// Check that we do not annotate deleting destructors with dead_on_return.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

class Foo {
public:
  virtual ~Foo();
};

// CHECK-LABEL: define dso_local void @_ZN3FooD0Ev
// CHECK-SAME: ptr noundef nonnull align 8 dereferenceable(8) [[THIS:%.*]]) unnamed_addr #[[ATTR0:[0-9]+]] align 2 {
Foo::~Foo() {};
