// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test() {
#pragma omp sections
  {
#pragma omp section
    ;
  }
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-section.c:3:1, line:9:1> line:3:6 test 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:13, line:9:1>
// CHECK-NEXT:     `-OMPSectionsDirective {{.*}} <line:4:1, col:21>
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:5:3, line:8:3>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:           |-CompoundStmt {{.*}} <line:5:3, line:8:3>
// CHECK-NEXT:           | `-OMPSectionDirective {{.*}} <line:6:1, col:20>
// CHECK-NEXT:           | `-NullStmt {{.*}} <line:7:5>
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-section.c:4:1) *const restrict'
