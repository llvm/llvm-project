// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefixes=CHECK-CXX

union Foo {
  struct Empty {} val;
};

union Foo foo = {};

// CHECK: @foo = {{.*}}global %union.Foo undef, align 1
// CHECK-CXX: @foo = {{.*}}global %union.Foo undef, align 1
