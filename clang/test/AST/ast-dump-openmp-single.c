// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test(void) {
#pragma omp single
  ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-single.c:3:1, line:6:1> line:3:6 test 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:6:1>
// CHECK-NEXT:     `-OMPSingleDirective {{.*}} <line:4:1, col:19>
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:5:3>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:           |-NullStmt {{.*}} <col:3>
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-single.c:4:1) *const restrict'
