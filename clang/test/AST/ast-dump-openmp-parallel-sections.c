// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_zero(void) {
#pragma omp parallel sections
  {}
}

void test_one(void) {
#pragma omp parallel sections
  { ; }
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: |-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-parallel-sections.c:3:1, line:6:1> line:3:6 test_zero 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:22, line:6:1>
// CHECK-NEXT: `-FunctionDecl {{.*}} <line:8:1, line:11:1> line:8:6 test_one 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:21, line:11:1>
// CHECK-NEXT:     `-OMPParallelSectionsDirective {{.*}} <line:9:1, col:30>
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:10:3, col:7>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:           |-CompoundStmt {{.*}} <col:3, col:7>
// CHECK-NEXT:           | `-NullStmt {{.*}} <col:5>
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <line:9:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-parallel-sections.c:9:1) *const restrict'
