// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test() {
#pragma omp parallel
  ;
}

// CHECK: TranslationUnitDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl 0x{{.*}} <{{.*}}ast-dump-openmp-parallel.c:3:1, line:6:1> line:3:6 test 'void ()'
// CHECK-NEXT:   `-CompoundStmt 0x{{.*}} <col:13, line:6:1>
// CHECK-NEXT:     `-OMPParallelDirective 0x{{.*}} <line:4:9, col:21>
// CHECK-NEXT:       `-CapturedStmt 0x{{.*}} <line:5:3>
// CHECK-NEXT:         `-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |-NullStmt 0x{{.*}} <col:3> openmp_structured_block
// CHECK-NEXT:           |-ImplicitParamDecl 0x{{.*}} <line:4:9> col:9 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl 0x{{.*}} <col:9> col:9 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:           `-ImplicitParamDecl 0x{{.*}} <col:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-parallel.c:4:9) *const restrict'
