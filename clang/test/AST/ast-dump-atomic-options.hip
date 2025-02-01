// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device %s \
// RUN:   -fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on \
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: FunctionDecl {{.*}} test_default
// CHECK-NOT: AttributedStmt
// CHECK-NOT: AtomicAttr
// CHECK: CompoundStmt
// CHECK-NEXT: `-AtomicExpr
__device__ __host__ void test_default(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: FunctionDecl {{.*}} test_one
// CHECK: `-AttributedStmt
// CHECK-NEXT: |-AtomicAttr {{.*}} no_remote_memory
// CHECK-NEXT: `-CompoundStmt
// CHECK-NEXT:   `-AtomicExpr
__device__ __host__ void test_one(float *a) {
  [[clang::atomic(no_remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK-LABEL: FunctionDecl {{.*}} test_two
// CHECK: `-AttributedStmt
// CHECK-NEXT: |-AtomicAttr {{.*}} !no_remote_memory ignore_denormal_mode
// CHECK-NEXT: `-CompoundStmt
// CHECK-NEXT:   `-AtomicExpr
__device__ __host__ void test_two(float *a) {
  [[clang::atomic(!no_remote_memory, ignore_denormal_mode)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK-LABEL: FunctionDecl {{.*}} test_three
// CHECK: `-AttributedStmt
// CHECK-NEXT: |-AtomicAttr {{.*}} no_remote_memory !no_fine_grained_memory !ignore_denormal_mode
// CHECK-NEXT: `-CompoundStmt
// CHECK-NEXT:   `-AtomicExpr
__device__ __host__ void test_three(float *a) {
  [[clang::atomic(no_remote_memory, !no_fine_grained_memory, !ignore_denormal_mode)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// CHECK-LABEL: FunctionDecl {{.*}} test_nested
// CHECK: CompoundStmt
// CHECK: |-AtomicExpr
// CHECK: `-AttributedStmt
// CHECK-NEXT: |-AtomicAttr {{.*}} !no_remote_memory !no_fine_grained_memory !ignore_denormal_mode
// CHECK-NEXT: `-CompoundStmt
// CHECK:     |-AtomicExpr
// CHECK:     |-AttributedStmt
// CHECK-NEXT:     |-AtomicAttr {{.*}} no_remote_memory
// CHECK-NEXT:     `-CompoundStmt
// CHECK-NEXT:       `-AtomicExpr
// CHECK:     `-AttributedStmt
// CHECK-NEXT:       |-AtomicAttr {{.*}} no_fine_grained_memory
// CHECK-NEXT:       `-CompoundStmt
// CHECK-NEXT:         `-AtomicExpr
__device__ __host__ void test_nested(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  [[clang::atomic(!no_remote_memory, !no_fine_grained_memory, !ignore_denormal_mode)]] {
    __scoped_atomic_fetch_max(a, 2, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
    [[clang::atomic(no_remote_memory)]] {
      __scoped_atomic_fetch_min(a, 3, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
    }
    [[clang::atomic(no_fine_grained_memory)]] {
      __scoped_atomic_fetch_sub(a, 4, __ATOMIC_RELEASE, __MEMORY_SCOPE_WVFRNT);
    }
  }
}

// CHECK-LABEL: FunctionTemplateDecl {{.*}} test_template
// CHECK: |-FunctionDecl {{.*}} test_template 'void (T *)'
// CHECK: | |-CompoundStmt
// CHECK: | | `-AttributedStmt
// CHECK: | |   |-AtomicAttr {{.*}} no_remote_memory
// CHECK: | |   `-CompoundStmt
// CHECK: | |     `-CallExpr {{.*}} '<dependent type>'
// CHECK: `-FunctionDecl {{.*}} used test_template 'void (float *)' implicit_instantiation
// CHECK:   |-CompoundStmt
// CHECK:   | `-AttributedStmt
// CHECK:   |   |-AtomicAttr {{.*}} no_remote_memory !no_fine_grained_memory !ignore_denormal_mode
// CHECK:   |   `-CompoundStmt
// CHECK:   |     `-AtomicExpr {{.*}} 'float'
template<typename T>
__device__ __host__ void test_template(T *a) {
  [[clang::atomic(no_remote_memory, !no_fine_grained_memory, !ignore_denormal_mode)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_template_caller() {
  float *p;
  test_template(p);
}
