// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20 -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

template <class T> __device__ T value;
template <int> struct Result {};

// Parentheses change an entity query into an expression query. Both templates
// must therefore have different mangled names even though normal expression
// mangling ignores parentheses.
template <class T>
__device__ Result<__addrspaceof(value<T>)> query(T) {
  return {};
}

template <class T>
__device__ Result<__addrspaceof((value<T>))> query(T) {
  return {};
}

template __device__ Result<__CLANG_ADDRESS_SPACE_CUDA_DEVICE> query<int>(int);
template __device__ Result<__CLANG_ADDRESS_SPACE_DEFAULT> query<int>(int);

// Lb1E records the entity form; Lb0E records the expression form.
// CHECK-DAG: define {{.*}}@_Z5queryIiE6ResultIXu13__addrspaceofLb1EX5valueIT_EEEEES1_(
// CHECK-DAG: define {{.*}}@_Z5queryIiE6ResultIXu13__addrspaceofLb0EX5valueIT_EEEEES1_(
