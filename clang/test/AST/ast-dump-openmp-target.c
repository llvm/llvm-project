// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck %s

void test_target(void) {
#pragma omp target
  ;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target
// CHECK:          `-OMPTargetDirective
// CHECK-NEXT:       `-CapturedStmt
// CHECK-NEXT:         `-CapturedDecl
// CHECK-NEXT:           |-CapturedStmt
// CHECK-NEXT:           | `-CapturedDecl
// CHECK-NEXT:           |   |-NullStmt
// CHECK-NEXT:           |   |-ImplicitParamDecl
// CHECK-NEXT:           |   `-ImplicitParamDecl
// CHECK-NEXT:           |-AlwaysInlineAttr
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-RecordDecl
// CHECK-NEXT:           | `-CapturedRecordAttr
// CHECK-NEXT:           `-CapturedDecl
// CHECK-NEXT:             |-NullStmt
// CHECK-NEXT:             |-ImplicitParamDecl
// CHECK-NEXT:             `-ImplicitParamDecl

void test_target_data(int x) {
#pragma omp target data map(x)
  ;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target_data
// CHECK:          `-OMPTargetDataDirective
// CHECK-NEXT:       |-OMPMapClause
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK-NEXT:       `-CapturedStmt
// CHECK-NEXT:         `-CapturedDecl
// CHECK-NEXT:           |-NullStmt
// CHECK-NEXT:           `-ImplicitParamDecl

void test_target_enter_data(int x) {
#pragma omp target enter data map(to \
                                  : x)
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target_enter_data
// CHECK:          `-OMPTargetEnterDataDirective {{.*}} openmp_standalone_directive
// CHECK-NEXT:       |-OMPMapClause
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK-NEXT:       `-CapturedStmt
// CHECK-NEXT:         `-CapturedDecl
// CHECK-NEXT:           |-CompoundStmt
// CHECK-NEXT:           |-AlwaysInlineAttr
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           `-ImplicitParamDecl

void test_target_exit_data(int x) {
#pragma omp target exit data map(from \
                                 : x)
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target_exit_data
// CHECK:          `-OMPTargetExitDataDirective {{.*}} openmp_standalone_directive
// CHECK-NEXT:       |-OMPMapClause
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} 'x' 'int'
// CHECK-NEXT:       `-CapturedStmt
// CHECK-NEXT:         `-CapturedDecl
// CHECK-NEXT:           |-CompoundStmt
// CHECK-NEXT:           |-AlwaysInlineAttr
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           `-ImplicitParamDecl

void test_target_parallel_for_simd(int x, int y) {
#pragma omp target parallel for simd
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target_parallel_for_simd
// CHECK:          `-OMPTargetParallelForSimdDirective
// CHECK-NEXT:       |-OMPFirstprivateClause
// CHECK-NEXT:       | |-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:       `-CapturedStmt
// CHECK-NEXT:         |-CapturedDecl
// CHECK-NEXT:         | |-CapturedStmt
// CHECK-NEXT:         | | |-CapturedDecl
// CHECK-NEXT:         | | | |-CapturedStmt
// CHECK-NEXT:         | | | | |-CapturedDecl
// CHECK-NEXT:         | | | | | |-ForStmt
// CHECK:              | | | | | | |   `-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              | | | | | | `-ForStmt
// CHECK:              | | | | | |   |   `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              | | | | | |-ImplicitParamDecl
// CHECK-NEXT:         | | | | | |-ImplicitParamDecl
// CHECK-NEXT:         | | | | | |-ImplicitParamDecl
// CHECK-NEXT:         | | | | | |-VarDecl
// CHECK-NEXT:         | | | | | | `-IntegerLiteral
// CHECK-NEXT:         | | | | | `-VarDecl
// CHECK-NEXT:         | | | | |   `-IntegerLiteral
// CHECK-NEXT:         | | | | |-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | | | | `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | | | |-ImplicitParamDecl
// CHECK-NEXT:         | | | |-ImplicitParamDecl
// CHECK-NEXT:         | | | |-RecordDecl
// CHECK-NEXT:         | | | | |-CapturedRecordAttr {{.*}} Implicit
// CHECK-NEXT:         | | | | |-FieldDecl {{.*}} implicit 'int'
// CHECK-NEXT:         | | | | | `-OMPCaptureKindAttr {{.*}} Implicit
// CHECK-NEXT:         | | | | `-FieldDecl {{.*}} implicit 'int'
// CHECK-NEXT:         | | | |   `-OMPCaptureKindAttr {{.*}} Implicit
// CHECK-NEXT:         | | | `-CapturedDecl
// CHECK-NEXT:         | | |   |-ForStmt
// CHECK:              | | |   | |   `-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              | | |   | `-ForStmt
// CHECK:              | | |   |   |   `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              | | |   |-ImplicitParamDecl
// CHECK-NEXT:         | | |   |-ImplicitParamDecl
// CHECK-NEXT:         | | |   |-ImplicitParamDecl
// CHECK-NEXT:         | | |   |-VarDecl
// CHECK-NEXT:         | | |   | `-IntegerLiteral
// CHECK-NEXT:         | | |   `-VarDecl
// CHECK-NEXT:         | | |     `-IntegerLiteral
// CHECK-NEXT:         | | |-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | | `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | |-AlwaysInlineAttr
// CHECK-NEXT:         | |-ImplicitParamDecl
// CHECK-NEXT:         | |-ImplicitParamDecl
// CHECK-NEXT:         | |-ImplicitParamDecl
// CHECK-NEXT:         | |-ImplicitParamDecl
// CHECK-NEXT:         | |-ImplicitParamDecl
// CHECK-NEXT:         | |-ImplicitParamDecl
// CHECK-NEXT:         | |-RecordDecl
// CHECK-NEXT:         | | |-CapturedRecordAttr {{.*}} Implicit
// CHECK-NEXT:         | | |-FieldDecl {{.*}} implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr {{.*}} Implicit
// CHECK-NEXT:         | | `-FieldDecl {{.*}} implicit 'int'
// CHECK-NEXT:         | |   `-OMPCaptureKindAttr {{.*}} Implicit
// CHECK-NEXT:         | `-CapturedDecl
// CHECK-NEXT:         |   |-CapturedStmt
// CHECK-NEXT:         |   | |-CapturedDecl
// CHECK-NEXT:         |   | | |-ForStmt
// CHECK:              |   | | | |   `-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              |   | | | `-ForStmt
// CHECK:              |   | | |   |   `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              |   | | |-ImplicitParamDecl
// CHECK-NEXT:         |   | | |-ImplicitParamDecl
// CHECK-NEXT:         |   | | |-ImplicitParamDecl
// CHECK-NEXT:         |   | | |-VarDecl
// CHECK-NEXT:         |   | | | `-IntegerLiteral
// CHECK-NEXT:         |   | | `-VarDecl
// CHECK-NEXT:         |   | |   `-IntegerLiteral
// CHECK-NEXT:         |   | |-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         |   | `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         |   |-ImplicitParamDecl
// CHECK-NEXT:         |   |-ImplicitParamDecl
// CHECK-NEXT:         |   |-RecordDecl
// CHECK-NEXT:         |   | |-CapturedRecordAttr {{.*}} Implicit
// CHECK-NEXT:         |   | |-FieldDecl {{.*}} implicit 'int'
// CHECK-NEXT:         |   | | `-OMPCaptureKindAttr {{.*}} Implicit
// CHECK-NEXT:         |   | `-FieldDecl {{.*}} implicit 'int'
// CHECK-NEXT:         |   |   `-OMPCaptureKindAttr {{.*}} Implicit
// CHECK-NEXT:         |   `-CapturedDecl
// CHECK-NEXT:         |     |-ForStmt
// CHECK:              |     | |   `-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              |     | `-ForStmt
// CHECK:              |     |   |   `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:              |     |-ImplicitParamDecl
// CHECK-NEXT:         |     |-ImplicitParamDecl
// CHECK-NEXT:         |     |-ImplicitParamDecl
// CHECK-NEXT:         |     |-VarDecl
// CHECK-NEXT:         |     | `-IntegerLiteral
// CHECK-NEXT:         |     `-VarDecl
// CHECK-NEXT:         |       `-IntegerLiteral
// CHECK-NEXT:         |-DeclRefExpr {{.*}} ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         `-DeclRefExpr {{.*}} ParmVar {{.*}} 'y' 'int'

void test_target_parallel_for_simd_collapse(int x, int y) {
#pragma omp target parallel for simd collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target_parallel_for_simd_collapse
// CHECK:          `-OMPTargetParallelForSimdDirective
// CHECK-NEXT:       |-OMPCollapseClause
// CHECK-NEXT:       | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       | |-value: Int 1
// CHECK-NEXT:       |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       |-OMPFirstprivateClause
// CHECK-NEXT:       | |-DeclRefExpr {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:       `-CapturedStmt
