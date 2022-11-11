// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

int x[100];
int y[100];

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
