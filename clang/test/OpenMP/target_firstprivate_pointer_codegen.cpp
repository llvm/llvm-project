// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

/// ========================================================================
/// Test: Firstprivate pointer handling in OpenMP target regions
/// ========================================================================
///
/// This test verifies that pointers with firstprivate semantics get the
/// OMP_MAP_LITERAL flag, enabling the runtime to pass pointer values directly
/// without performing present table lookups.
///
/// Map type values:
///   288 = OMP_MAP_TARGET_PARAM (32) + OMP_MAP_LITERAL (256)
///         Used for explicit firstprivate(ptr)
///
///   800 = OMP_MAP_TARGET_PARAM (32) + OMP_MAP_LITERAL (256) + OMP_MAP_IS_PTR (512)
///         Used for implicit firstprivate pointers (e.g., from defaultmap clauses)
///         Note: 512 is OMP_MAP_IS_PTR, not IMPLICIT. Implicitness is tracked separately.
///
///   544 = OMP_MAP_TARGET_PARAM (32) + OMP_MAP_IS_PTR (512)
///         Incorrect behavior - missing LITERAL flag, causes runtime present table lookup
///

///==========================================================================
/// Test 1: Explicit firstprivate(pointer) → map type 288
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{[^.]*}} = private unnamed_addr constant [1 x i64] [i64 288]
// CHECK-DAG: @.offload_sizes{{[^.]*}} = private unnamed_addr constant [1 x i64] zeroinitializer

void test1_explicit_firstprivate() {
  double *ptr = nullptr;
  
  // Explicit firstprivate should generate map type 288
  // (TARGET_PARAM | LITERAL, no IS_PTR flag for explicit clauses)
  #pragma omp target firstprivate(ptr)
  {
    if (ptr) ptr[0] = 1.0;
  }
}

///==========================================================================
/// Test 2: defaultmap(firstprivate:pointer) → map type 800
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] zeroinitializer

void test2_defaultmap_firstprivate_pointer() {
  double *ptr = nullptr;
  
  // defaultmap(firstprivate:pointer) creates implicit firstprivate
  // Should generate map type 800 (TARGET_PARAM | LITERAL | IS_PTR)
  #pragma omp target defaultmap(firstprivate:pointer)
  {
    if (ptr) ptr[0] = 2.0;
  }
}

///==========================================================================
/// Test 3: defaultmap(firstprivate:scalar) with double → map type 800
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 800]

void test3_defaultmap_scalar_double() {
  double d = 3.0;
  
  // OpenMP's "scalar" category excludes pointers but includes arithmetic types
  // Double gets implicit firstprivate → map type 800
  #pragma omp target defaultmap(firstprivate:scalar)
  {
    d += 1.0;
  }
}

///==========================================================================
/// Test 4: Pointer with defaultmap(firstprivate:scalar) → map type 800
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] zeroinitializer

void test4_pointer_with_scalar_defaultmap() {
  double *ptr = nullptr;
  
  // Note: defaultmap(firstprivate:scalar) does NOT apply to pointers (scalar excludes pointers).
  // However, the pointer still gets 800 because in OpenMP 5.0+, pointers without explicit
  // data-sharing attributes are implicitly firstprivate and lowered as IS_PTR|LITERAL|TARGET_PARAM.
  // This is the default pointer behavior, NOT due to the scalar defaultmap.
  #pragma omp target defaultmap(firstprivate:scalar)
  {
    if (ptr) ptr[0] = 4.0;
  }
}

///==========================================================================
/// Test 5: Multiple pointers with explicit firstprivate → all get 288
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 288, i64 288, i64 288]
// CHECK-DAG: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] zeroinitializer

void test5_multiple_firstprivate() {
  int *a = nullptr;
  float *b = nullptr;
  double *c = nullptr;
  
  // All explicit firstprivate pointers get map type 288
  #pragma omp target firstprivate(a, b, c)
  {
    if (a) a[0] = 6;
    if (b) b[0] = 7.0f;
    if (c) c[0] = 8.0;
  }
}

///==========================================================================
/// Test 6: Pointer to const with firstprivate → map type 288
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 288]
// CHECK-DAG: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] zeroinitializer

void test6_const_pointer() {
  const double *const_ptr = nullptr;
  
  // Const pointer with explicit firstprivate → 288
  #pragma omp target firstprivate(const_ptr)
  {
    if (const_ptr) {
      double val = const_ptr[0];
      (void)val;
    }
  }
}

///==========================================================================
/// Test 7: Pointer-to-pointer with firstprivate → map type 288
///==========================================================================

// CHECK-DAG: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 288]
// CHECK-DAG: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] zeroinitializer

void test7_pointer_to_pointer() {
  int **pp = nullptr;
  
  // Pointer-to-pointer with explicit firstprivate → 288
  #pragma omp target firstprivate(pp)
  {
    if (pp && *pp) (*pp)[0] = 9;
  }
}

///==========================================================================
/// Verification: The key fix is that firstprivate pointers now include
/// the LITERAL flag (256), which tells the runtime to pass the pointer
/// value directly instead of performing a present table lookup.
///
/// Before fix: Pointers got 544 (TARGET_PARAM | IS_PTR) → runtime lookup
/// After fix:  Pointers get 288 or 800 (includes LITERAL) → direct pass
///==========================================================================

#endif // HEADER
