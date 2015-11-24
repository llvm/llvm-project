// RUN: %clang_cc1 %s -O0 -ffake-address-space-map -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O0 -DCL20 -cl-std=CL2.0 -ffake-address-space-map -emit-llvm -o - | FileCheck %s --check-prefix=CL20

// CHECK: i32* %arg
void f__p(__private int *arg) {}

// CHECK: i32 addrspace(1)* %arg
void f__g(__global int *arg) {}

// CHECK: i32 addrspace(2)* %arg
void f__l(__local int *arg) {}

// CHECK: i32 addrspace(3)* %arg
void f__c(__constant int *arg) {}

// CHECK: i32* %arg
void fp(private int *arg) {}

// CHECK: i32 addrspace(1)* %arg
void fg(global int *arg) {}

// CHECK: i32 addrspace(2)* %arg
void fl(local int *arg) {}

// CHECK: i32 addrspace(3)* %arg
void fc(constant int *arg) {}

#ifdef CL20
int i;
// CL20-DAG: @i = common addrspace(1) global i32 0
int *ptr;
// CL20-DAG: @ptr = common addrspace(1) global i32 addrspace(4)* null
#endif

// CHECK: i32* %arg
// CL20-DAG: i32 addrspace(4)* %arg
void f(int *arg) {

  int i;
// CHECK: %i = alloca i32,
// CL20-DAG: %i = alloca i32,

#ifdef CL20
  static int ii;
// CL20-DAG: @f.ii = internal addrspace(1) global i32 0
#endif
}
