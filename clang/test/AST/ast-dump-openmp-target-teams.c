// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test(void) {
#pragma omp target teams
  ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-target-teams.c:3:1, line:6:1> line:3:6 test 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:17, line:6:1>
// CHECK-NEXT:     `-OMPTargetTeamsDirective {{.*}} <line:4:1, col:25>
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:5:3>
// CHECK-NEXT:         `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |-CapturedStmt {{.*}} <col:3>
// CHECK-NEXT:           | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |   |-CapturedStmt {{.*}} <col:3>
// CHECK-NEXT:           |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |   |   |-NullStmt {{.*}} <col:3>
// CHECK-NEXT:           |   |   |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:           |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:           |   |   `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
// CHECK-NEXT:           |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
// CHECK-NEXT:           |   |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:           |   | `-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:           |   `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:           |     |-NullStmt {{.*}} <line:5:3>
// CHECK-NEXT:           |     |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:           |     |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:           |     `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
// CHECK-NEXT:           |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT:           |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
// CHECK-NEXT:           |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:           | `-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:           `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:             |-CapturedStmt {{.*}} <line:5:3>
// CHECK-NEXT:             | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:             |   |-NullStmt {{.*}} <col:3>
// CHECK-NEXT:             |   |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:             |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:             |   `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
// CHECK-NEXT:             |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
// CHECK-NEXT:             |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:             | `-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:             `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:               |-NullStmt {{.*}} <line:5:3>
// CHECK-NEXT:               |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:               |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:               `-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-target-teams.c:4:1) *const restrict'
