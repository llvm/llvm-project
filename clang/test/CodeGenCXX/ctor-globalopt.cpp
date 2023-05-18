// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s -O2 | opt - -S -passes=globalopt -o - | FileCheck %s --check-prefix=O1
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -o - %s -O2 | opt - -S -passes=globalopt -o - | FileCheck %s --check-prefix=O1

// Check that GlobalOpt can eliminate static constructors for simple implicit
// constructors. This is a targeted integration test to make sure that LLVM's
// optimizers are able to process Clang's IR. GlobalOpt in particular is
// sensitive to the casts we emit.

// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] 
// CHECK: [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_ctor_globalopt.cpp, ptr null }]

// CHECK-LABEL: define internal {{.*}}void @_GLOBAL__sub_I_ctor_globalopt.cpp()
// CHECK: call {{.*}}void @
// CHECK-NOT: call{{ }}

// O1: @llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

struct A {
  virtual void f();
  int a;
};
struct B : virtual A {
  virtual void g();
  int b;
};
B b;
