// RUN: %clang_cc1 -cl-std=c++ %s -emit-llvm -o - -O0 -triple spir-unknown-unknown | FileCheck %s

template <typename T>
struct S{
  T a;
  T foo();
};

template<typename T>
T S<T>::foo() { return a;}

//CHECK: %struct.S = type { i32 }
//CHECK: %struct.S.0 = type { i32 addrspace(4)* }
//CHECK: %struct.S.1 = type { i32 addrspace(1)* }

//CHECK: i32 @_ZN1SIiE3fooEv(%struct.S* %this)
//CHECK: i32 addrspace(4)* @_ZN1SIPU3AS4iE3fooEv(%struct.S.0* %this)
//CHECK: i32 addrspace(1)* @_ZN1SIPU3AS1iE3fooEv(%struct.S.1* %this)

void bar(){
  S<int> sint;
  S<int*> sintptr;
  S<__global int*> sintptrgl;

  sint.foo();
  sintptr.foo();
  sintptrgl.foo();
}
