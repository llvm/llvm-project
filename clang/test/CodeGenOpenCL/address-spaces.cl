// RUN: %clang_cc1 %s -O0 -ffake-address-space-map -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,SPIR
// RUN: %clang_cc1 %s -O0 -DCL20 -cl-std=CL2.0 -ffake-address-space-map -emit-llvm -o - | FileCheck %s --check-prefixes=CL20,CL20SPIR
// RUN: %clang_cc1 %s -O0 -triple amdgcn-amd-amdhsa-opencl -emit-llvm -o - | FileCheck --check-prefixes=CHECK,SPIR %s
// RUN: %clang_cc1 %s -O0 -triple amdgcn-amd-amdhsa-opencl -DCL20 -cl-std=CL2.0 -emit-llvm -o - | FileCheck %s --check-prefixes=CL20,CL20SPIR
// RUN: %clang_cc1 %s -O0 -triple amdgcn-amd-amdhsa-amdgizcl -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,GIZ
// RUN: %clang_cc1 %s -O0 -triple amdgcn-amd-amdhsa-amdgizcl -DCL20 -cl-std=CL2.0 -emit-llvm -o - | FileCheck %s --check-prefixes=CL20,CL20GIZ

// SPIR: i32* %arg
// GIZ: i32 addrspace(5)* %arg
void f__p(__private int *arg) {}

// CHECK: i32 addrspace(1)* %arg
void f__g(__global int *arg) {}

// CHECK: i32 addrspace(3)* %arg
void f__l(__local int *arg) {}

// CHECK: i32 addrspace(2)* %arg
void f__c(__constant int *arg) {}

// SPIR: i32* %arg
// GIZ: i32 addrspace(5)* %arg
void fp(private int *arg) {}

// CHECK: i32 addrspace(1)* %arg
void fg(global int *arg) {}

// CHECK: i32 addrspace(3)* %arg
void fl(local int *arg) {}

// CHECK: i32 addrspace(2)* %arg
void fc(constant int *arg) {}

#ifdef CL20
int i;
// CL20-DAG: @i = common addrspace(1) global i32 0
int *ptr;
// CL20SPIR-DAG: @ptr = common addrspace(1) global i32 addrspace(4)* null
// CL20GIZ-DAG: @ptr = common addrspace(1) global i32* null
#endif

// SPIR: i32* %arg
// GIZ: i32 addrspace(5)* %arg
// CL20SPIR-DAG: i32 addrspace(4)* %arg
// CL20GIZ-DAG: i32* %arg
void f(int *arg) {

  int i;
// SPIR: %i = alloca i32,
// GIZ: %i = alloca i32{{.*}}addrspace(5)
// CL20SPIR-DAG: %i = alloca i32,
// CL20GIZ-DAG: %i = alloca i32{{.*}}addrspace(5)

#ifdef CL20
  static int ii;
// CL20-DAG: @f.ii = internal addrspace(1) global i32 0
#endif
}
