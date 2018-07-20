// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck %s

#define __private__ __attribute__((address_space(5)))

void func_pchar(__private__ char *x);

void test_cast(char *gen_ptr) {
  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  __private__ char *priv_ptr = (__private__ char *)gen_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pcharPU3AS5c(i8 addrspace(5)* %[[cast]])
  func_pchar((__private__ char *)gen_ptr);
}
