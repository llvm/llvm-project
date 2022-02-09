// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test(void) {
#pragma omp barrier
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-barrier.c:3:1, line:5:1> line:3:6 test 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:5:1>
// CHECK-NEXT:     `-OMPBarrierDirective {{.*}} <line:4:1, col:20> openmp_standalone_directive
