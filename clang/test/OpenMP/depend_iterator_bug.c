// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

int x[100];
int y[100];
int z[100][100];

// CHECK-LABEL: @many_iterators_single_clause(
// CHECK:    [[VLA:%.*]] = alloca [[STRUCT_KMP_DEPEND_INFO:%.*]], i64 10, align 16
// CHECK:    = call i32 @__kmpc_omp_task_with_deps(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 10, ptr {{.*}}, i32 0, ptr null)
void many_iterators_single_clause(void) {
    #pragma omp task depend(iterator(j=0:5), in: x[j], y[j])
    {
    }
}

// CHECK-LABEL: @many_iterators_many_clauses(
// CHECK:    [[VLA:%.*]] = alloca [[STRUCT_KMP_DEPEND_INFO:%.*]], i64 10, align 16
// CHECK:    = call i32 @__kmpc_omp_task_with_deps(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 10, ptr {{.*}}, i32 0, ptr null)
void many_iterators_many_clauses(void) {
    #pragma omp task depend(iterator(j=0:5), in: x[j]) \
                     depend(iterator(j=0:5), in: y[j])
    {
    }
}

// CHECK-LABEL: @multidim_iterators_clause1(
// CHECK:    [[VLA:%.*]] = alloca [[STRUCT_KMP_DEPEND_INFO:%.*]], i64 1, align 16
// CHECK:    = call i32 @__kmpc_omp_task_with_deps(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 1, ptr {{.*}}, i32 0, ptr null)
void multidim_iterators_clause1(void) {
    #pragma omp task depend(iterator(i=0:1, j=0:1), in: z[i][j])
    {
    }
}

// CHECK-LABEL: @multidim_iterators_offset_clause(
// CHECK:    [[VLA:%.*]] = alloca [[STRUCT_KMP_DEPEND_INFO:%.*]], i64 1, align 16
// CHECK:    = call i32 @__kmpc_omp_task_with_deps(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 1, ptr {{.*}}, i32 0, ptr null)
void multidim_iterators_offset_clause(void) {
    #pragma omp task depend(iterator(i=5:6, j=10:11), in: z[i][j])
    {
    }
}

// CHECK-LABEL: @multidim_iterators_clause25(
// CHECK:    [[VLA:%.*]] = alloca [[STRUCT_KMP_DEPEND_INFO:%.*]], i64 25, align 16
// CHECK:    = call i32 @__kmpc_omp_task_with_deps(ptr {{.*}}, i32 {{.*}}, ptr {{.*}}, i32 25, ptr {{.*}}, i32 0, ptr null)
void multidim_iterators_clause25(void) {
    #pragma omp task depend(iterator(i=0:5, j=0:5), in: z[i][j])
    {
    }
}

