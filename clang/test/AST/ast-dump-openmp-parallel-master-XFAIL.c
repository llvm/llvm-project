// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -fopenmp-version=50 -ast-dump %s 2>&1 | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

// REQUIRES: broken-PR41022
// https://bugs.llvm.org/show_bug.cgi?id=41022

void test_zero() {
#pragma omp parallel master
  ;
}
// CHECK: {{.*}}ast-dump-openmp-parallel-master-XFAIL.c:4:22: warning: extra tokens at the end of '#pragma omp parallel' are ignored

void test_one() {
#pragma omp parallel master
  { ; }
}
// CHECK: {{.*}}ast-dump-openmp-parallel-master-XFAIL.c:10:22: warning: extra tokens at the end of '#pragma omp parallel' are ignored

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: |-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-parallel-master-XFAIL.c:3:1, line:6:1> line:3:6 test_zero 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:18, line:6:1>
// CHECK-NEXT: |   `-OMPParallelDirective {{.*}} <line:4:9, col:28>
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:5:3>
// CHECK-NEXT: |       `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |         |-NullStmt {{.*}} <col:3>
// CHECK-NEXT: |         |-ImplicitParamDecl {{.*}} <line:4:9> col:9 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |         |-ImplicitParamDecl {{.*}} <col:9> col:9 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |         `-ImplicitParamDecl {{.*}} <col:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-parallel-master-XFAIL.c:4:9) *const restrict'
// CHECK-NEXT: `-FunctionDecl {{.*}} <line:9:1, line:12:1> line:9:6 test_one 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:12:1>
// CHECK-NEXT:     `-OMPParallelDirective {{.*}} <line:10:9, col:28>
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:11:3, col:7>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |-CompoundStmt {{.*}} <col:3, col:7>
// CHECK-NEXT:           | `-NullStmt {{.*}} <col:5>
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <line:10:9> col:9 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:9> col:9 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <col:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-parallel-master-XFAIL.c:10:9) *const restrict'
