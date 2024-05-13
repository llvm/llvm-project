// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test(void) {
#pragma omp flush
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-flush.c:3:1, line:5:1> line:3:6 test 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:5:1>
// CHECK-NEXT:     `-OMPFlushDirective {{.*}} <line:4:1, col:18> openmp_standalone_directive
