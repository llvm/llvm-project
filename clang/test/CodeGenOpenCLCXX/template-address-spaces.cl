// RUN: %clang_cc1 -cl-std=c++ %s -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck %s

template <typename T>
struct S{
  T a;
  T foo();
};

template<typename T>
T S<T>::foo() { return a;}

// CHECK: %struct.S = type { i32 }
// CHECK: %struct.S.0 = type { i32 addrspace(4)* }
// CHECK: %struct.S.1 = type { i32 addrspace(1)* }

// CHECK:  %0 = addrspacecast %struct.S* %sint to %struct.S addrspace(4)*
// CHECK:  %call = call i32 @_ZNU3AS41SIiE3fooEv(%struct.S addrspace(4)* %0) #1
// CHECK:  %1 = addrspacecast %struct.S.0* %sintptr to %struct.S.0 addrspace(4)*
// CHECK:  %call1 = call i32 addrspace(4)* @_ZNU3AS41SIPU3AS4iE3fooEv(%struct.S.0 addrspace(4)* %1) #1
// CHECK:  %2 = addrspacecast %struct.S.1* %sintptrgl to %struct.S.1 addrspace(4)*
// CHECK:  %call2 = call i32 addrspace(1)* @_ZNU3AS41SIPU3AS1iE3fooEv(%struct.S.1 addrspace(4)* %2) #1

void bar(){
  S<int> sint;
  S<int*> sintptr;
  S<__global int*> sintptrgl;

  sint.foo();
  sintptr.foo();
  sintptrgl.foo();
}
