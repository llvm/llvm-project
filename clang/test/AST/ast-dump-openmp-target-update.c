// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test(int x) {
#pragma omp target update to(x)
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-target-update.c:3:1, line:5:1> line:3:6 test 'void (int)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:11, col:15> col:15 used x 'int'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:18, line:5:1>
// CHECK-NEXT:     `-OMPTargetUpdateDirective {{.*}} <line:4:1, col:32> openmp_standalone_directive
// CHECK-NEXT:       |-OMPToClause {{.*}} <col:27, col:31>
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:       `-CapturedStmt {{.*}} <col:1>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |-CompoundStmt {{.*}} <col:1>
// CHECK-NEXT:           |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-update.c:4:1) *const restrict'
