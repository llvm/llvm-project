// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck --check-prefix=HOST %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -o - %s \
// RUN:   | FileCheck --check-prefix=DEVICE %s

#include "Inputs/cuda.h"

__host__ __device__ int hd();
int hd() { return 42; }

struct S {
  __host__ __device__ int f();
};
int S::f() { return 42; }

template <typename T>
struct T1 {
  __host__ __device__ int f();
};
template <typename T>
int T1<T>::f() { return 42; }

__global__ void kernel(int *out) {
  S s;
  T1<int> t;
  *out = hd() + s.f() + t.f();
}

int host_caller() {
  S s;
  T1<int> t;
  return hd() + s.f() + t.f();
}

// HOST-DAG: define{{.*}} i32 @_Z2hdv()
// HOST-DAG: define{{.*}} i32 @_ZN1S1fEv(
// HOST-DAG: define{{.*}} i32 @_ZN2T1IiE1fEv(

// DEVICE-DAG: define{{.*}} i32 @_Z2hdv()
// DEVICE-DAG: define{{.*}} i32 @_ZN1S1fEv(
// DEVICE-DAG: define{{.*}} i32 @_ZN2T1IiE1fEv(
