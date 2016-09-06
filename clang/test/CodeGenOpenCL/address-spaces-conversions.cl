// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -ffake-address-space-map -cl-std=CL2.0 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -cl-std=CL2.0 -emit-llvm -o - | FileCheck --check-prefix=CHECK-NOFAKE %s
// When -ffake-address-space-map is not used, all addr space mapped to 0 for x86_64.

// test that we generate address space casts everywhere we need conversions of
// pointers to different address spaces

// CHECK: define void @test
void test(global int *arg_glob, generic int *arg_gen) {
  int var_priv;
  arg_gen = arg_glob; // implicit cast global -> generic
  // CHECK: %{{[0-9]+}} = addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(4)*
  // CHECK-NOFAKE-NOT: addrspacecast

  arg_gen = &var_priv; // implicit cast with obtaining adr, private -> generic
  // CHECK: %{{[0-9]+}} = addrspacecast i32* %var_priv to i32 addrspace(4)*
  // CHECK-NOFAKE-NOT: addrspacecast

  arg_glob = (global int *)arg_gen; // explicit cast
  // CHECK: %{{[0-9]+}} = addrspacecast i32 addrspace(4)* %{{[0-9]+}} to i32 addrspace(1)*
  // CHECK-NOFAKE-NOT: addrspacecast

  global int *var_glob =
      (global int *)arg_glob; // explicit cast in the same address space
  // CHECK-NOT: %{{[0-9]+}} = addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(1)*
  // CHECK-NOFAKE-NOT: addrspacecast

  var_priv = arg_gen - arg_glob; // arithmetic operation
  // CHECK: %{{.*}} = ptrtoint i32 addrspace(4)* %{{.*}} to i64
  // CHECK: %{{.*}} = ptrtoint i32 addrspace(1)* %{{.*}} to i64
  // CHECK-NOFAKE: %{{.*}} = ptrtoint i32* %{{.*}} to i64
  // CHECK-NOFAKE: %{{.*}} = ptrtoint i32* %{{.*}} to i64

  var_priv = arg_gen > arg_glob; // comparison
  // CHECK: %{{[0-9]+}} = addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(4)*

  generic void *var_gen_v = arg_glob;
  // CHECK: addrspacecast
  // CHECK-NOT: bitcast
  // CHECK-NOFAKE: bitcast
  // CHECK-NOFAKE-NOT: addrspacecast
}

// Test ternary operator.
// CHECK: define void @test_ternary
void test_ternary(void) {
  global int *var_glob;
  generic int *var_gen;
  generic int *var_gen2;
  generic float *var_gen_f;
  generic void *var_gen_v;

  var_gen = var_gen ? var_gen : var_gen2; // operands of the same addr spaces and the same type
  // CHECK: icmp
  // CHECK-NOT: addrspacecast
  // CHECK-NOT: bitcast
  // CHECK: phi
  // CHECK: store i32 addrspace(4)* %{{.+}}, i32 addrspace(4)** %{{.+}}

  var_gen = var_gen ? var_gen : var_glob; // operands of overlapping addr spaces and the same type
  // CHECK: icmp
  // CHECK-NOT: bitcast
  // CHECK: %{{.+}} = addrspacecast i32 addrspace(1)* %{{.+}} to i32 addrspace(4)*
  // CHECK: phi
  // CHECK: store

  typedef int int_t;
  global int_t *var_glob_typedef;
  var_gen = var_gen ? var_gen : var_glob_typedef; // operands of overlapping addr spaces and equivalent types
  // CHECK: icmp
  // CHECK-NOT: bitcast
  // CHECK: %{{.+}} = addrspacecast i32 addrspace(1)* %{{.+}} to i32 addrspace(4)*
  // CHECK: phi
  // CHECK: store
 
  var_gen_v = var_gen ? var_gen : var_gen_f; // operands of the same addr space and different types
  // CHECK: icmp
  // CHECK: %{{.+}} = bitcast i32 addrspace(4)* %{{.+}} to i8 addrspace(4)*
  // CHECK: %{{.+}} = bitcast float addrspace(4)* %{{.+}} to i8 addrspace(4)*
  // CHECK: phi
  // CHECK: store

  var_gen_v = var_gen ? var_glob : var_gen_f; // operands of overlapping addr spaces and different types
  // CHECK: icmp
  // CHECK: %{{.+}} = addrspacecast i32 addrspace(1)* %{{.+}} to i8 addrspace(4)*
  // CHECK: %{{.+}} = bitcast float addrspace(4)* %{{.+}} to i8 addrspace(4)*
  // CHECK: phi
  // CHECK: store
}
