// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -o - | FileCheck %s
// PR7291

struct Foo {
  unsigned file_id;

  Foo(unsigned arg);
};

Foo::Foo(unsigned arg) : file_id(arg = 42)
{ }

// CHECK: define {{.*}} @_ZN3FooC2Ej(ptr {{[^,]*}} %this, i32 noundef %arg) unnamed_addr
// CHECK: [[ARG:%.*]] = alloca i32
// CHECK: store i32 42, ptr [[ARG]]
// CHECK: store i32 42, ptr %{{.*}}
// CHECK: ret {{void|%struct.Foo}}
