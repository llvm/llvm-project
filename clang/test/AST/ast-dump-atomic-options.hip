// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fcuda-is-device %s \
// RUN:   -fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on \
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: FunctionDecl {{.*}} test_default
// CHECK: | |-CompoundStmt
// CHECK-NOT: AtomicNoRemoteMemory
// CHECK-NOT: AtomicNoFineGrainedMemory
// CHECK-NOT: AtomicIgnoreDenormalMode
__device__ __host__ void test_default(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: FunctionDecl {{.*}} test_one
// CHECK: | |-CompoundStmt {{.*}} AtomicNoRemoteMemory=1
// CHECK-NOT: AtomicNoFineGrainedMemory
// CHECK-NOT: AtomicIgnoreDenormalMode
__device__ __host__ void test_one(float *a) {
  #pragma clang atomic no_remote_memory(on)
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: FunctionDecl {{.*}} test_two
// CHECK: | |-CompoundStmt {{.*}} AtomicNoRemoteMemory=0 AtomicIgnoreDenormalMode=1
// CHECK-NOT: AtomicNoFineGrainedMemory
__device__ __host__ void test_two(float *a) {
  #pragma clang atomic no_remote_memory(off) ignore_denormal_mode(on)
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: FunctionDecl {{.*}} test_three
// CHECK: | |-CompoundStmt {{.*}} AtomicNoRemoteMemory=1 AtomicNoFineGrainedMemory=0 AtomicIgnoreDenormalMode=0
__device__ __host__ void test_three(float *a) {
  #pragma clang atomic no_remote_memory(on) no_fine_grained_memory(off) ignore_denormal_mode(off)
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: FunctionDecl {{.*}} test_nested
// CHECK: | |-CompoundStmt
// CHECK-NOT: AtomicNoRemoteMemory
// CHECK-NOT: AtomicNoFineGrainedMemory
// CHECK-NOT: AtomicIgnoreDenormalMode
// CHECK: | | `-CompoundStmt {{.*}} AtomicNoRemoteMemory=0 AtomicNoFineGrainedMemory=0 AtomicIgnoreDenormalMode=0
// CHECK: | |   |-CompoundStmt {{.*}} AtomicNoRemoteMemory=1 AtomicNoFineGrainedMemory=0 AtomicIgnoreDenormalMode=0
// CHECK: | |   `-CompoundStmt {{.*}} AtomicNoRemoteMemory=0 AtomicNoFineGrainedMemory=1 AtomicIgnoreDenormalMode=0
__device__ __host__ void test_nested(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  {
    #pragma clang atomic no_remote_memory(off) no_fine_grained_memory(off) ignore_denormal_mode(off)
    __scoped_atomic_fetch_max(a, 2, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
    {
      #pragma clang atomic no_remote_memory(on)
      __scoped_atomic_fetch_min(a, 3, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
    }
    {
      #pragma clang atomic no_fine_grained_memory(on)
      __scoped_atomic_fetch_sub(a, 4, __ATOMIC_RELEASE, __MEMORY_SCOPE_WVFRNT);
    }
  }
}

// CHECK-LABEL: FunctionTemplateDecl {{.*}} test_template
// CHECK-LABEL: FunctionDecl {{.*}} test_template 'void (T *)'
// CHECK: | |-CompoundStmt {{.*}} AtomicNoRemoteMemory=1 AtomicNoFineGrainedMemory=0 AtomicIgnoreDenormalMode=0
// CHECK-LABEL: FunctionDecl {{.*}} used test_template 'void (float *)' implicit_instantiation
// CHECK: | |-CompoundStmt {{.*}} AtomicNoRemoteMemory=1 AtomicNoFineGrainedMemory=0 AtomicIgnoreDenormalMode=0
template<typename T>
__device__ __host__ void test_template(T *a) {
  #pragma clang atomic no_remote_memory(on) no_fine_grained_memory(off) ignore_denormal_mode(off)
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

__device__ __host__ void test_template_caller() {
  float *p;
  test_template(p);
}
