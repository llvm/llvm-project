// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test() {
#pragma omp master
  ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-master.c:3:1, line:6:1> line:3:6 test 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:13, line:6:1>
// CHECK-NEXT:     `-OMPMasterDirective {{.*}} <line:4:1, col:19>
// CHECK-NEXT:       `-NullStmt {{.*}} <line:5:3>
