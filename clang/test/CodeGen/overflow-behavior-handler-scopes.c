// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fopenmp -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OMP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fopenmp-simd -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OMP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -std=c23 -fdefer-ts -Wno-redundant-defer -DDEFER -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DEFER

// Checks only reach the handler from code written in the function body.

// A check in a variable length array size would skip the allocation.
// CHECK-LABEL: define {{.*}} @vla_size(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap
int vla_size(int __ob_trap n) {
  int v[n + 1];
  v[0] = 0;
  return v[0];
__overflow:
  return -1;
}

// A check in an OpenMP region cannot leave the structured block, even when
// the region is emitted inline.
// OMP-LABEL: define {{.*}} @omp_simd(
// OMP: @llvm.sadd.with.overflow.i32
// OMP-NOT: br label %__overflow
// OMP: call void @llvm.ubsantrap
int omp_simd(int __ob_trap a, int n) {
  #pragma omp simd
  for (int i = 0; i < n; i++)
    a = a + i;
  return a;
__overflow:
  return -1;
}

#ifdef DEFER
// The body of a defer statement runs as a cleanup, on every path out of its
// scope.
// DEFER-LABEL: define {{.*}} @defer_body(
// DEFER: @llvm.sadd.with.overflow.i32
// DEFER-NOT: br label %__overflow
// DEFER: call void @llvm.ubsantrap
int defer_body(int __ob_trap a) {
  {
    _Defer { a = a + 1; }
  }
  return a;
__overflow:
  return -1;
}
#endif
