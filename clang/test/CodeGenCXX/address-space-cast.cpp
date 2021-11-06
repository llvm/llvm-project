// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck %s

#define __private__ __attribute__((address_space(5)))

void func_pchar(__private__ char *x);
void func_pvoid(__private__ void *x);
void func_pint(__private__ int *x);

class Base {
};

class Derived : public Base {
};

void fn(Derived *p) {
  __private__ Base *b = (__private__ Base *)p;
}

void test_cast(char *gen_char_ptr, void *gen_void_ptr, int *gen_int_ptr) {
  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  __private__ char *priv_char_ptr = (__private__ char *)gen_char_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  priv_char_ptr = (__private__ char *)gen_void_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  priv_char_ptr = (__private__ char *)gen_int_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  __private__ void *priv_void_ptr = (__private__ void *)gen_char_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  priv_void_ptr = (__private__ void *)gen_void_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: store i8 addrspace(5)* %[[cast]]
  priv_void_ptr = (__private__ void *)gen_int_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i32 addrspace(5)*
  // CHECK-NEXT: store i32 addrspace(5)* %[[cast]]
  __private__ int *priv_int_ptr = (__private__ int *)gen_void_ptr;

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pcharPU3AS5c(i8 addrspace(5)* noundef %[[cast]])
  func_pchar((__private__ char *)gen_char_ptr);

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pcharPU3AS5c(i8 addrspace(5)* noundef %[[cast]])
  func_pchar((__private__ char *)gen_void_ptr);

  // CHECK: %[[cast:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pcharPU3AS5c(i8 addrspace(5)* noundef %[[cast]])
  func_pchar((__private__ char *)gen_int_ptr);

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pvoidPU3AS5v(i8 addrspace(5)* noundef %[[cast]])
  func_pvoid((__private__ void *)gen_char_ptr);

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pvoidPU3AS5v(i8 addrspace(5)* noundef %[[cast]])
  func_pvoid((__private__ void *)gen_void_ptr);

  // CHECK: %[[cast:.*]] = addrspacecast i32* %{{.*}} to i8 addrspace(5)*
  // CHECK-NEXT: call void @_Z10func_pvoidPU3AS5v(i8 addrspace(5)* noundef %[[cast]])
  func_pvoid((__private__ void *)gen_int_ptr);

  // CHECK: %[[cast:.*]] = addrspacecast i8* %{{.*}} to i32 addrspace(5)*
  // CHECK-NEXT: call void @_Z9func_pintPU3AS5i(i32 addrspace(5)* noundef %[[cast]])
  func_pint((__private__ int *)gen_void_ptr);
}
