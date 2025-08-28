// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang %s -std=c++20 -fenable-ripple -Xclang -disable-llvm-passes -Wripple -S -emit-llvm -o - | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>

// We expect to see the same number of initialization variables as there are parallel loops!

// CHECK: foo
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
void foo(size_t n, float *C, float *A, float *B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  ripple_parallel(BS, 0, 1);
  for (int i = 0; i < n; i = i + 192)
    C[i] = A[i] + B[i];
  ripple_parallel(BS, 0, 1);
  for (int i = 0; i < n; i = i + 192)
    C[i] = A[i] + B[i];
  ripple_parallel(BS, 0, 1);
  for (int i = 0; i < n; i = i + 192)
    C[i] = A[i] + B[i];
}

struct C {
  // CHECK: baz
  // CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  // CHECK: bal
  // CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
  void bal(size_t n, float *C, float *A, float *B) {
    ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
    ripple_parallel(BS, 0, 1);
    for (int i = 0; i < n; i = i + 192)
      C[i] = A[i] + B[i];
  }
  void baz(size_t n, float *C, float *A, float *B);
};

void C::baz(size_t n, float *C, float *A, float *B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
  ripple_parallel(BS, 0, 1);
  for (int i = 0; i < n; i = i + 192)
    C[i] = A[i] + B[i];
  bal(n, C, A, B);
}

template <typename T>
struct TC {
  void bip(size_t n, T *C, T *A, T *B) {
    ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
    ripple_parallel(BS, 0, 1);
    for (int i = 0; i < n; i = i + 192)
      C[i] = A[i] + B[i];
  }

  void bap(size_t n, T *C, T *A, T *B);
};

template<typename T>
void TC<T>::bap(size_t n, T *C, T *A, T *B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

  {
  ripple_parallel(BS, 0);
  for (int i = 0; i < n; i = i + 192)
    C[i] = A[i] + B[i];
  }
  auto baplambda = [=]() {
    ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
    ripple_parallel(BS, 1);
    for (int i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];
  };
  baplambda();
  ripple_parallel(BS, 1);
  for (int i = 0; i < n; i = i + 192)
    C[i] += A[i] + B[i];
  baplambda();
  ripple_parallel(BS, 0, 1);
  for (int i = 0; i < n; i = i + 192)
    C[i] += A[i] + B[i];
}

// CHECK: bip
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: bap
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// The lambda in bap
// CHECK: bap
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
template class TC<int>;
// CHECK: bip
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: bap
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// The lambda in bap
// CHECK: bap
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
template class TC<float>;
// CHECK: bip
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: bap
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// The lambda in bap
// CHECK: bap
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
template class TC<unsigned>;

template <typename T, typename IT>
struct T2 {
  void bop(IT n, T *C, T *A, T *B) {
    ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

    ripple_parallel(BS, 0);
    for (IT i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];
    {
    ripple_parallel(BS, 0);
    for (IT i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];
    }
    ripple_parallel(BS, 0);
    for (IT i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];

    auto x = [=]<typename OtherT>() {
      ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
      ripple_parallel(BS, 1);
      for (OtherT i = 0; i < n; i = i + 33)
        C[i] += A[i] + B[i];
    };
    x.template operator()<T>();

    ripple_parallel(BS, 0);
    for (IT i = 0; i < n; i++)
      C[i] += A[i] + B[i];
  }
  void pop(IT n, T *C, T *A, T *B);
};

template<typename T, typename IT>
void T2<T, IT>::pop(IT n, T *C, T *A, T *B) {
  ripple_block_t BS = ripple_set_block_shape(0, 4, 8);

  auto x = [=]<typename OtherT>() {
    ripple_block_t BS = ripple_set_block_shape(0, 4, 8);
    ripple_parallel(BS, 1);
    for (OtherT i = 0; i < n; i = i + 33)
      C[i] += A[i] + B[i];
    ripple_parallel(BS, 0);
    for (OtherT i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];
    {
      ripple_parallel(BS, 0, 1);
      for (OtherT i = 0; i < n; i = i + 192)
        C[i] += A[i] + B[i];
    }
    ripple_parallel(BS, 0);
    for (OtherT i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];
    ripple_parallel(BS, 1);
    for (OtherT i = 0; i < n; i = i + 192)
      C[i] += A[i] + B[i];
  };

  x.template operator()<T>();
}

// CHECK: bop
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// Lambda in bop
// CHECK: bop
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: pop
// Lambda in pop
// CHECK: pop
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
template class T2<int, short>;
// CHECK: bop
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// Lambda in bop
// CHECK: bop
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: pop
// Lambda in pop
// CHECK: pop
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
// CHECK: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}
template class T2<unsigned, signed>;

// CHECK-NOT: ripple.par.iv{{[0-9]*}} = alloca i{{[0-9]+}}