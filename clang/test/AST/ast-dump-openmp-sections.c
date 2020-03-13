// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_zero() {
#pragma omp sections
  {}
}

void test_one() {
#pragma omp sections
  { ; }
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: |-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-sections.c:3:1, line:6:1> line:3:6 test_zero 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:18, line:6:1>
// CHECK-NEXT: `-FunctionDecl {{.*}} <line:8:1, line:11:1> line:8:6 test_one 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:11:1>
// CHECK-NEXT:     `-OMPSectionsDirective {{.*}} <line:9:1, col:21>
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:10:3, col:7>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:           |-CompoundStmt {{.*}} <col:3, col:7>
// CHECK-NEXT:           | `-NullStmt {{.*}} <col:5>
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <line:9:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-sections.c:9:1) *const restrict'
