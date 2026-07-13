// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu \
// RUN: -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// RUN %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu \
// RUN -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

// Tests that const-qualified aggregates without mutable members are implicitly
// mapped as 'to' instead of 'tofrom' under defaultmap(tofrom:aggregate) and
// explicit map clauses. Structs that have mutable members, or that are
// non-const, must continue to be mapped 'tofrom'.

struct foo {
  foo(int j) : i(j) {};
  int i;
};

struct foo_mutable {
  foo_mutable(int j) : i(j), m(0) {};
  int i;
  mutable int m;
};

// TODO: A const foo_mutable should ideally only copy back its mutable
// member 'm' and ignore non-mutable member 'i' on a 'from' mapping, per
// OpenMP 6.0 p299 lines 3-4. This requires per-member mapper generation
// and is left for a follow-up patch.
struct foo_nested {
  foo_nested(int j) : inner(j), z(j) {};
  foo inner;
  const int z;
};

struct foo_nested_mutable {
  foo_nested_mutable(int j) : inner(j), z(j) {};
  foo_mutable inner; // has mutable member buried inside
  const int z;
};

// CHECK: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 545, i64 288]
// CHECK: @.offload_maptypes.2 = private unnamed_addr constant [2 x i64] [i64 547, i64 288]
// CHECK: @.offload_maptypes.4 = private unnamed_addr constant [2 x i64] [i64 547, i64 288]
// CHECK: @.offload_maptypes.6 = private unnamed_addr constant [2 x i64] [i64 545, i64 288]
// CHECK: @.offload_maptypes.8 = private unnamed_addr constant [2 x i64] [i64 547, i64 288]
// CHECK: @.offload_maptypes.10 = private unnamed_addr constant [2 x i64] [i64 545, i64 288]
// CHECK: @.offload_maptypes.12 = private unnamed_addr constant [2 x i64] [i64 33, i64 288]
// CHECK: @.offload_maptypes.14 = private unnamed_addr constant [2 x i64] [i64 32, i64 288]
// CHECK: @.offload_maptypes.16 = private unnamed_addr constant [2 x i64] [i64 33, i64 288]
// CHECK: @.offload_maptypes.18 = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK: @.offload_maptypes.20 = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK: @.offload_maptypes.22 = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK: @.offload_maptypes.24 = private unnamed_addr constant [3 x i64] [i64 545, i64 547, i64 288]
// CHECK: @.offload_maptypes.26 = private unnamed_addr constant [2 x i64] [i64 545, i64 288]
// CHECK: @.offload_maptypes.28 = private unnamed_addr constant [1 x i64] [i64 2]

// ---------------------------------------------------------------------------
// Implicit mapping tests (no explicit map clause, defaultmap governs)
// ---------------------------------------------------------------------------

// Const struct with no mutable members. Mapped as TO|TARGET_PARAM|IMPLICIT = 545.
// LABEL: test_const_no_mutable
// CHECK: store ptr @.offload_maptypes, ptr {{.*}}, align 8
void test_const_no_mutable() {
  const foo a(2);
#pragma omp target
  {
    int x = a.i;
  }
}

// Non-const struct. Mapped as TO|FROM|TARGET_PARAM|IMPLICIT = 547.
// LABEL: define dso_local void @_Z13test_nonconstv
// CHECK: store ptr @.offload_maptypes.2, ptr {{.*}}, align 8
void test_nonconst() {
  foo a(2);
#pragma omp target
  {
    int x = a.i;
  }
}

// Const struct with a mutable member. Mapped as TO|FROM|TARGET_PARAM|IMPLICIT = 547.
// LABEL: define dso_local void @_Z23test_const_with_mutablev
// CHECK: store ptr @.offload_maptypes.4, ptr {{.*}}, align 8
void test_const_with_mutable() {
  const foo_mutable a(2);
#pragma omp target
  {
    a.m = 1;
  }
}

// Const struct whose members are themselves all const and free of mutable
// fields. Mapped as TO|TARGET_PARAM|IMPLICIT = 545.
// LABEL: define dso_local void @_Z17test_const_nestedv() #0 {
// CHECK: store ptr @.offload_maptypes.6, ptr {{.*}}, align 8
void test_const_nested() {
  const foo_nested a(2);
#pragma omp target
  {
    int x = a.inner.i;
  }
}

// Const array of a const-qualified struct type.
// Mapped as TO|FROM|TARGET_PARAM|IMPLICIT = 547.
// LABEL: define dso_local void @_Z30test_const_nested_with_mutablev
// CHECK: store ptr @.offload_maptypes.8, ptr {{.*}}, align 8
void test_const_nested_with_mutable() {
  const foo_nested_mutable a(2);
#pragma omp target
  {
    a.inner.m = 1;
  }
}

// Const array of a const-qualified struct type.
// Mapped as TO|TARGET_PARAM|IMPLICIT = 545.
// LABEL: define dso_local void @_Z16test_const_arrayv
// CHECK: store ptr @.offload_maptypes.10, ptr {{.*}}, align 8
void test_const_array() {
  const foo arr[4] = {1, 2, 3, 4};
#pragma omp target
  {
    int x = arr[0].i;
  }
}

// ---------------------------------------------------------------------------
// Explicit map clause tests
// ---------------------------------------------------------------------------

// Explicit map(tofrom:) on a const struct. Mapped as TO|TARGET_PARAM = 33.
// LABEL: define dso_local void @_Z27test_explicit_tofrom_const
// CHECK: store ptr @.offload_maptypes.12, ptr {{.*}}, align 8
void test_explicit_tofrom_const() {
  const foo a(2);
#pragma omp target map(tofrom:a)
  {
    int x = a.i;
  }
}

// Explicit map(from:) on a const struct. The FROM clause is ignored.
// Mapped as TARGET_PARAM = 32.
// LABEL: define dso_local void @_Z24test_explicit_from_constv
// CHECK: store ptr @.offload_maptypes.14, ptr {{.*}}, align 8
void test_explicit_from_const() {
  const foo a(2);
#pragma omp target map(from:a)
  {
    int x = a.i;
  }
}

// Explicit map(to:) on a const struct. Mapped as TO|TARGET_PARAM = 33.
// LABEL: define dso_local void @_Z22test_explicit_to_constv()
// CHECK: store ptr @.offload_maptypes.16, ptr {{.*}}, align 8
void test_explicit_to_const() {
  const foo a(2);
#pragma omp target map(to:a)
  {
    int x = a.i;
  }
}

// ---------------------------------------------------------------------------
// target update from tests
// ---------------------------------------------------------------------------

// target update from on a const struct with no mutable members. The FROM clause
// is ignored. Mapped as FROM = 2.
// LABEL: define dso_local void @_Z29test_target_update_from_constv
// CHECK:   call void @__tgt_target_data_update_mapper(ptr @1, i64 -1, i32 1, ptr %3, ptr %4, ptr @.offload_sizes.17, ptr @.offload_maptypes.18, ptr null, ptr null)
void test_target_update_from_const() {
  const foo a(2);
#pragma omp target update from(a)
}

// target update from on a non-const struct. Mapped as FROM = 2.
// LABEL: define dso_local void @_Z32test_target_update_from_nonconstv
// CHECK: call void @__tgt_target_data_update_mapper(ptr @1, i64 -1, i32 1, ptr %3, ptr %4, ptr @.offload_sizes.19, ptr @.offload_maptypes.20, ptr null, ptr null)
void test_target_update_from_nonconst() {
  foo a(2);
#pragma omp target update from(a)
}

// target update from on a const struct that has a mutable member. Mapped as FROM = 2.
// LABEL: define dso_local void @_Z37test_target_update_from_const_mutablev
// CHECK: call void @__tgt_target_data_update_mapper(ptr @1, i64 -1, i32 1, ptr %3, ptr %4, ptr @.offload_sizes.21, ptr @.offload_maptypes.22, ptr null, ptr null)
void test_target_update_from_const_mutable() {
  const foo_mutable a(2);
#pragma omp target update from(a)
}

// ---------------------------------------------------------------------------
// Combined tests
// ---------------------------------------------------------------------------

// Mixed region with one const and one non-const variable of the same struct
// type. Each variable gets its own map type: const maps as
// TO|TARGET_PARAM|IMPLICIT = 545, non-const maps as
// TO|FROM|TARGET_PARAM|IMPLICIT = 547.
// LABEL: define dso_local void @_Z10test_mixedv
// CHECK: store ptr @.offload_maptypes.24, ptr {{.*}}, align 8
void test_mixed() {
  const foo ca(2);
  foo ma(3);
#pragma omp target
  {
    int x = ca.i;
    ma.i = 99;
  }
}

// Explicit defaultmap(tofrom:aggregate) directive on a const struct.
// Mapped as TO|TARGET_PARAM|IMPLICIT = 545.
// LABEL: define dso_local void @_Z31test_defaultmap_tofrom_explicitv
// CHECK: store ptr @.offload_maptypes.26, ptr {{.*}}, align 8
void test_defaultmap_tofrom_explicit() {
  const foo a(2);
#pragma omp target defaultmap(tofrom:aggregate)
  {
    int x = a.i;
  }
}

// User-defined mapper on const struct — FROM must NOT be suppressed because the
// mapper accesses non-const pointee data py[0:10].
// Mapped as FROM = 2.
// LABEL: define dso_local void @_Z30test_user_defined_mapper_constv
// CHECK: call void @__tgt_target_data_update_mapper(ptr @1, i64 -1, i32 1, ptr {{.*}}, ptr {{.*}}, ptr @.offload_sizes.27, ptr @.offload_maptypes.28, ptr null, ptr {{.*}})
int y[10];
struct S {
  int x;
  int *py;
};

#pragma omp declare mapper(m1: const S s) map(alloc: s.x, s.py) map(from: s.py[0:10])

void test_user_defined_mapper_const() {
  int data[10] = {0};
  const S s1 = {1, data};
  #pragma omp target update from(mapper(m1): s1)
}
