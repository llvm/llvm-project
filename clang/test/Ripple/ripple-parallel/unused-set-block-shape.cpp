// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - 2>&1 | FileCheck %s

#include <ripple.h>

// CHECK-NOT: warning: unused variable 'BS'

template<typename T>
void test1(size_t start, size_t end, T *x, T *y, T *xpy) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  ripple_parallel(BS, 0);
  for (size_t i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}

template<> void test1(size_t, size_t, float*, float*, float*);
template<> void test1(size_t, size_t, int*, int*, int*);

void test2(size_t start, size_t end, char *x, char *y, char *xpy) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  ripple_parallel(BS, 0);
  for (size_t i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}

template<typename T>
void test3(size_t start, size_t end, T *x, T *y, T *xpy) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  ripple_parallel(BS, 0);
  for (size_t i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}
