//RUN: %clang_cc1 %s -triple spir -cl-std=clc++ -emit-llvm -O0 -o - | FileCheck %s

//CHECK-LABEL: define{{.*}} spir_func void @_Z3barPU3AS1i
void bar(global int *gl) {
  //CHECK: addrspacecast i32 addrspace(1)* %{{[0-9]+}} to i32 addrspace(4)*
  int *gen = addrspace_cast<int *>(gl);
}
