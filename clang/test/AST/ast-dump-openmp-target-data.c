// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test(int x) {
#pragma omp target data map(x)
  ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-target-data.c:3:1, line:6:1> line:3:6 test 'void (int)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:11, col:15> col:15 used x 'int'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:18, line:6:1>
// CHECK-NEXT:     `-OMPTargetDataDirective {{.*}} <line:4:1, col:31>
// CHECK-NEXT:       |-OMPMapClause {{.*}} <col:25, col:30>
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} <col:29> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:5:3>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:           |-NullStmt {{.*}} <col:3>
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-data.c:4:1) *const restrict'
