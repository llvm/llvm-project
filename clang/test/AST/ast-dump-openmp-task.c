// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -fopenmp-version=50 -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

typedef unsigned long omp_event_handle_t;
void test(void) {
  omp_event_handle_t evt;
#pragma omp task detach(evt)
  ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <line:4:1, line:8:1> line:4:6 test 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:8:1>
// CHECK:          `-OMPTaskDirective {{.*}} <line:6:1, col:29>
// CHECK-NEXT:       |-OMPDetachClause {{.+}} <col:18, col:28>
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} <col:25> 'omp_event_handle_t':'unsigned long' lvalue Var {{.+}} 'evt' 'omp_event_handle_t':'unsigned long'
// CHECK-NEXT:       |-OMPFirstprivateClause {{.+}} <<invalid sloc>> <implicit>
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} <col:25> 'omp_event_handle_t':'unsigned long' lvalue Var {{.+}} 'evt' 'omp_event_handle_t':'unsigned long'
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:7:3>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |-NullStmt {{.*}} <col:3>
// CHECK-NEXT:           |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <line:6:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT:           `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-task.c:6:1) *const restrict'
