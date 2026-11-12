// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fexceptions \
// RUN:   -o - -x hip %s | FileCheck %s

#include "Inputs/cuda.h"

int* hvar;
__device__ int* dvar;

// CHECK-LABEL: define {{.*}}@_Znwm
// CHECK:    load ptr, ptr @hvar
void* operator new(unsigned long size) {
  return hvar;
}
// CHECK-LABEL: define {{.*}}@_ZdlPv
// CHECK:    store ptr inttoptr (i64 1 to ptr), ptr @hvar
void operator delete(void *p) {
  hvar = (int*)1;
}

__device__ void* operator new(unsigned long size) {
  return dvar;
}

__device__ void operator delete(void *p) {
  dvar = (int*)11;
}

class A {
  int x;
public:
  A(){
    x = 123;
  }
};

template<class T>
class shared_ptr {
   int id;
   T *ptr;
public:
  shared_ptr(T *p) {
    id = 2;
    ptr = p;
  }
};

// The constructor of B calls the delete operator to clean up
// the memory allocated by the new operator when exceptions happen.
// Make sure the host delete operator is used on host side.
//
// No need to do similar checks on the device side since it does
// not support exception.

// CHECK-LABEL: define {{.*}}@main
// CHECK:    call void @_ZN1BC1Ev

// CHECK-LABEL: define {{.*}}@_ZN1BC1Ev
// CHECK:    call void @_ZN1BC2Ev

// CHECK-LABEL: define {{.*}}@_ZN1BC2Ev
// CHECK: call {{.*}}@_Znwm
// CHECK:  invoke void @_ZN1AC1Ev
// CHECK:  call void @_ZN10shared_ptrI1AEC1EPS0_
// CHECK:  cleanup
// CHECK:  call void @_ZdlPv

struct B{
  shared_ptr<A> pa{new A};
};

int main() {
  B b;
}
