// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(int x) {
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
#pragma omp target
#pragma omp teams distribute parallel for simd collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
#pragma omp target
#pragma omp teams distribute parallel for simd collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
#pragma omp target
#pragma omp teams distribute parallel for simd collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: |-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:3:1, line:8:1> line:3:6 test_one 'void (int)'
// CHECK: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK: | `-CompoundStmt {{.*}} <col:22, line:8:1>
// CHECK: |   `-OMPTargetDirective {{.*}} <line:4:1, col:19>
// CHECK: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK: |     | `-DeclRefExpr {{.*}} <line:6:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt {{.*}} <line:5:1, col:47>
// CHECK: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt {{.*}} <col:1, col:47>
// CHECK: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:47>
// CHECK: |       | | | | `-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       | | | |   | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | | |-<<<NULL>>>
// CHECK: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       | | | |   | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | |   | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK: |       | | | |   |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   | |-<<<NULL>>>
// CHECK: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       | | | |   |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |     |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK: |       | | | |-RecordDecl {{.*}} <line:5:1> col:1 implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-CapturedStmt {{.*}} <col:3, line:7:5>
// CHECK: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       | | | | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       | | | | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK: |       | | | |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | |-<<<NULL>>>
// CHECK: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       | | | |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |     |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |-OMPCapturedExprDecl {{.*}} <col:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK: |       | | |     |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       | | |     | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       | | |     | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       | | |     | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | |     | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit .global_tid. 'const int'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:5:1, col:47>
// CHECK: |       |   | `-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       |   |   | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | | |-<<<NULL>>>
// CHECK: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       |   |   | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   |   | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK: |       |   |   |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   | |-<<<NULL>>>
// CHECK: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       |   |   |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |     |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK: |       |   |-RecordDecl {{.*}} <line:5:1> col:1 implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-CapturedStmt {{.*}} <col:3, line:7:5>
// CHECK: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK: |       |   | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | | |-<<<NULL>>>
// CHECK: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       |   | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK: |       |   |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | |-<<<NULL>>>
// CHECK: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | `-NullStmt {{.*}} <line:7:5>
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK: |       |   |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |     |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |-OMPCapturedExprDecl {{.*}} <col:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK: |       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       |       | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       |       | |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: |-FunctionDecl {{.*}} <line:10:1, line:16:1> line:10:6 test_two 'void (int, int)'
// CHECK: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK: | |-ParmVarDecl {{.*}} <col:22, col:26> col:26 used y 'int'
// CHECK: | `-CompoundStmt {{.*}} <col:29, line:16:1>
// CHECK: |   `-OMPTargetDirective {{.*}} <line:11:1, col:19>
// CHECK: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK: |     | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt {{.*}} <line:12:1, col:47>
// CHECK: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt {{.*}} <col:1, col:47>
// CHECK: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:47>
// CHECK: |       | | | | `-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | |   | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | | |-<<<NULL>>>
// CHECK: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       | | | |   | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK: |       | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       | | | |   | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | |   | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK: |       | | | |   | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | |   |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   | |-<<<NULL>>>
// CHECK: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       | | | |   |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |   |   |-<<<NULL>>>
// CHECK: |       | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       | | | |   |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl {{.*}} <line:11:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK: |       | | | |-RecordDecl {{.*}} <line:12:1> col:1 implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK: |       | | | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       | | | | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | | | | |   |-<<<NULL>>>
// CHECK: |       | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       | | | | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK: |       | | | | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       | | | |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | |-<<<NULL>>>
// CHECK: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       | | | |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |   |-<<<NULL>>>
// CHECK: |       | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       | | | |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |-OMPCapturedExprDecl {{.*}} <line:13:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK: |       | | |     |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       | | |     | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       | | |     | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       | | |     | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | |     | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl {{.*}} <line:11:1> col:1 implicit .global_tid. 'const int'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int'
// CHECK: |       | | | `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:12:1, col:47>
// CHECK: |       |   | `-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   |   | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | | |-<<<NULL>>>
// CHECK: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       |   |   | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   | | | |   |-<<<NULL>>>
// CHECK: |       |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       |   |   | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   |   | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK: |       |   |   | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   |   |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   | |-<<<NULL>>>
// CHECK: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       |   |   |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |   |   |-<<<NULL>>>
// CHECK: |       |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       |   |   |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl {{.*}} <line:11:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK: |       |   |-RecordDecl {{.*}} <line:12:1> col:1 implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK: |       |   | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | | |-<<<NULL>>>
// CHECK: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       |   | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   | | | |   |-<<<NULL>>>
// CHECK: |       |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       |   | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK: |       |   | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK: |       |   |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | |-<<<NULL>>>
// CHECK: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | `-ForStmt {{.*}} <line:14:5, line:15:7>
// CHECK: |       |   |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK: |       |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |   |-<<<NULL>>>
// CHECK: |       |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK: |       |   |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |-OMPCapturedExprDecl {{.*}} <line:13:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK: |       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       |       | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       |       | |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: |       `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK: |-FunctionDecl {{.*}} <line:18:1, line:24:1> line:18:6 test_three 'void (int, int)'
// CHECK: | |-ParmVarDecl {{.*}} <col:17, col:21> col:21 used x 'int'
// CHECK: | |-ParmVarDecl {{.*}} <col:24, col:28> col:28 used y 'int'
// CHECK: | `-CompoundStmt {{.*}} <col:31, line:24:1>
// CHECK: |   `-OMPTargetDirective {{.*}} <line:19:1, col:19>
// CHECK: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK: |     | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt {{.*}} <line:20:1, col:59>
// CHECK: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt {{.*}} <col:1, col:59>
// CHECK: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:59>
// CHECK: |       | | | | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK: |       | | | | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK: |       | | | | | |-value: Int 1
// CHECK: |       | | | | |   `-IntegerLiteral {{.*}} <col:57> 'int' 1
// CHECK: |       | | | | `-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | |   | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | | |-<<<NULL>>>
// CHECK: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       | | | |   | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK: |       | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       | | | |   | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | |   | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK: |       | | | |   | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | |   |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   | |-<<<NULL>>>
// CHECK: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       | | | |   |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |   |   |-<<<NULL>>>
// CHECK: |       | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       | | | |   |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl {{.*}} <line:19:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK: |       | | | |-RecordDecl {{.*}} <line:20:1> col:1 implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK: |       | | | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       | | | | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | | | | |   |-<<<NULL>>>
// CHECK: |       | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       | | | | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK: |       | | | | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       | | | |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | |-<<<NULL>>>
// CHECK: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       | | | |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |   |-<<<NULL>>>
// CHECK: |       | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       | | | |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |-OMPCapturedExprDecl {{.*}} <line:21:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK: |       | | |     |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       | | |     | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       | | |     | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       | | |     | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | |     | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl {{.*}} <line:19:1> col:1 implicit .global_tid. 'const int'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int'
// CHECK: |       | | | `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:20:1, col:59>
// CHECK: |       |   | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK: |       |   | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK: |       |   | | |-value: Int 1
// CHECK: |       |   | |   `-IntegerLiteral {{.*}} <col:57> 'int' 1
// CHECK: |       |   | `-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   |   | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | | |-<<<NULL>>>
// CHECK: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       |   |   | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   | | | |   |-<<<NULL>>>
// CHECK: |       |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       |   |   | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   |   | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK: |       |   |   | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   |   |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   | |-<<<NULL>>>
// CHECK: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       |   |   |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |   |   |-<<<NULL>>>
// CHECK: |       |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       |   |   |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl {{.*}} <line:19:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK: |       |   |-RecordDecl {{.*}} <line:20:1> col:1 implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK: |       |   | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | | |-<<<NULL>>>
// CHECK: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       |   | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   | | | |   |-<<<NULL>>>
// CHECK: |       |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       |   | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK: |       |   | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK: |       |   |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | |-<<<NULL>>>
// CHECK: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | `-ForStmt {{.*}} <line:22:5, line:23:7>
// CHECK: |       |   |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK: |       |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |   |-<<<NULL>>>
// CHECK: |       |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK: |       |   |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |-OMPCapturedExprDecl {{.*}} <line:21:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK: |       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       |       | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       |       | |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: |       `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK: |-FunctionDecl {{.*}} <line:26:1, line:32:1> line:26:6 test_four 'void (int, int)'
// CHECK: | |-ParmVarDecl {{.*}} <col:16, col:20> col:20 used x 'int'
// CHECK: | |-ParmVarDecl {{.*}} <col:23, col:27> col:27 used y 'int'
// CHECK: | `-CompoundStmt {{.*}} <col:30, line:32:1>
// CHECK: |   `-OMPTargetDirective {{.*}} <line:27:1, col:19>
// CHECK: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK: |     | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt {{.*}} <line:28:1, col:59>
// CHECK: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt {{.*}} <col:1, col:59>
// CHECK: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:59>
// CHECK: |       | | | | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK: |       | | | | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK: |       | | | | | |-value: Int 2
// CHECK: |       | | | | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK: |       | | | | `-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | |   | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | | |-<<<NULL>>>
// CHECK: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       | | | |   | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK: |       | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | | |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       | | | |   | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | |   | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK: |       | | | |   | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | |   |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   | |-<<<NULL>>>
// CHECK: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       | | | |   |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |   |   |-<<<NULL>>>
// CHECK: |       | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       | | | |   |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl {{.*}} <line:27:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK: |       | | | |-RecordDecl {{.*}} <line:28:1> col:1 implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK: |       | | | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       | | | | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | | | | |   |-<<<NULL>>>
// CHECK: |       | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | | | | |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       | | | | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | | | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK: |       | | | | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       | | | |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   | |-<<<NULL>>>
// CHECK: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       | | | |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |   |   |-<<<NULL>>>
// CHECK: |       | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       | | | |   |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       | | | |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       | | | |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | | |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       | | | |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | | |-OMPCapturedExprDecl {{.*}} <line:29:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-OMPCapturedExprDecl {{.*}} <line:30:25> col:25 implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl {{.*}} <line:29:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK: |       | | |     |-BinaryOperator {{.*}} <col:3, line:30:28> 'long' '*'
// CHECK: |       | | |     | |-ImplicitCastExpr {{.*}} <line:29:3, col:26> 'long' <IntegralCast>
// CHECK: |       | | |     | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       | | |     | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | |   | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       | | |     | |   |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       | | |     | |   |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |   |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       | | |     | |   |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       | | |     | |   |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     | |   |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       | | |     | `-ImplicitCastExpr {{.*}} <line:30:5, col:28> 'long' <IntegralCast>
// CHECK: |       | | |     |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK: |       | | |     |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK: |       | | |     |     | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
// CHECK: |       | | |     |     |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       | | |     |     |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       | | |     |     |   `-ParenExpr {{.*}} <col:5> 'int'
// CHECK: |       | | |     |     |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     |     |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
// CHECK: |       | | |     |     |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       | | |     |     |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK: |       | | |     |     |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK: |       | | |     `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK: |       | | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl {{.*}} <line:27:1> col:1 implicit .global_tid. 'const int'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int'
// CHECK: |       | | | `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:28:1, col:59>
// CHECK: |       |   | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK: |       |   | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK: |       |   | | |-value: Int 2
// CHECK: |       |   | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK: |       |   | `-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   |   | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | | |-<<<NULL>>>
// CHECK: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       |   |   | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   | | | |   |-<<<NULL>>>
// CHECK: |       |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | | |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       |   |   | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   |   | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK: |       |   |   | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   |   |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   | |-<<<NULL>>>
// CHECK: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       |   |   |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |   |   |-<<<NULL>>>
// CHECK: |       |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       |   |   |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl {{.*}} <line:27:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK: |       |   |-RecordDecl {{.*}} <line:28:1> col:1 implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK: |       |   | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | | |-<<<NULL>>>
// CHECK: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       |   | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   | | | |   |-<<<NULL>>>
// CHECK: |       |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   | | | |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       |   | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK: |       |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK: |       |   | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK: |       |   | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK: |       |   |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   | |-<<<NULL>>>
// CHECK: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK: |       |   |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK: |       |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |   |   |-<<<NULL>>>
// CHECK: |       |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK: |       |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK: |       |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: |       |   |   |   `-NullStmt {{.*}} <line:31:7>
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK: |       |   |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK: |       |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |   |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK: |       |   |     |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |   |-OMPCapturedExprDecl {{.*}} <line:29:23> col:23 implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-OMPCapturedExprDecl {{.*}} <line:30:25> col:25 implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl {{.*}} <line:29:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK: |       |       |-BinaryOperator {{.*}} <col:3, line:30:28> 'long' '*'
// CHECK: |       |       | |-ImplicitCastExpr {{.*}} <line:29:3, col:26> 'long' <IntegralCast>
// CHECK: |       |       | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK: |       |       | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | |   | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK: |       |       | |   |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK: |       |       | |   |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       |       | |   |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK: |       |       | |   |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |   |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK: |       |       | |   |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK: |       |       | |   |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       | |   |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK: |       |       | `-ImplicitCastExpr {{.*}} <line:30:5, col:28> 'long' <IntegralCast>
// CHECK: |       |       |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK: |       |       |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK: |       |       |     | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
// CHECK: |       |       |     |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK: |       |       |     |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK: |       |       |     |   `-ParenExpr {{.*}} <col:5> 'int'
// CHECK: |       |       |     |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
// CHECK: |       |       |     |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
// CHECK: |       |       |     |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK: |       |       |     |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK: |       |       |     |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |       |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK: |       |       `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK: |       |         `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK: |       |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: |       `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK: `-FunctionDecl {{.*}} <line:34:1, line:41:1> line:34:6 test_five 'void (int, int, int)'
// CHECK:   |-ParmVarDecl {{.*}} <col:16, col:20> col:20 used x 'int'
// CHECK:   |-ParmVarDecl {{.*}} <col:23, col:27> col:27 used y 'int'
// CHECK:   |-ParmVarDecl {{.*}} <col:30, col:34> col:34 used z 'int'
// CHECK:   `-CompoundStmt {{.*}} <col:37, line:41:1>
// CHECK:     `-OMPTargetDirective {{.*}} <line:35:1, col:19>
// CHECK:       |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK:       | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:       | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:       | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:       `-CapturedStmt {{.*}} <line:36:1, col:59>
// CHECK:         |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | |-CapturedStmt {{.*}} <col:1, col:59>
// CHECK:         | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:59>
// CHECK:         | | | | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK:         | | | | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK:         | | | | | |-value: Int 2
// CHECK:         | | | | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK:         | | | | `-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |   | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |   | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | |   | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | |   | | | | |-<<<NULL>>>
// CHECK:         | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         | | | |   | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | |   | | | |   |-<<<NULL>>>
// CHECK:         | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         | | | |   | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         | | | |   | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | |   | | | |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | |   | | | |     |-<<<NULL>>>
// CHECK:         | | | |   | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         | | | |   | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         | | | |   | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         | | | |   | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         | | | |   | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | |   | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | |   | | | | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | |   | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | |   | | |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | |   | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK:         | | | |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         | | | |   | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK:         | | | |   | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK:         | | | |   | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK:         | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |   |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | |   |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | |   |   | |-<<<NULL>>>
// CHECK:         | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         | | | |   |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | |   |   |   |-<<<NULL>>>
// CHECK:         | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         | | | |   |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         | | | |   |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | |   |   |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | |   |   |     |-<<<NULL>>>
// CHECK:         | | | |   |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         | | | |   |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         | | | |   |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         | | | |   |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         | | | |   |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         | | | |   |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | |   |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | |   |   | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | |   |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | |   |     |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | |   |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |-ImplicitParamDecl {{.*}} <line:35:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK:         | | | |-RecordDecl {{.*}} <line:36:1> col:1 implicit struct definition
// CHECK:         | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         | | | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK:         | | | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK:         | | | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK:         | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | | | | | |-<<<NULL>>>
// CHECK:         | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         | | | | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | | | | |   |-<<<NULL>>>
// CHECK:         | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         | | | | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         | | | | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | | | | |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | | | | |     |-<<<NULL>>>
// CHECK:         | | | | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         | | | | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         | | | | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         | | | | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         | | | | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         | | | | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | | | | | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | | | |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK:         | | | | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         | | | | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK:         | | | | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK:         | | | | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK:         | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         | | | |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | |   | |-<<<NULL>>>
// CHECK:         | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         | | | |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | |   |   |-<<<NULL>>>
// CHECK:         | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         | | | |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         | | | |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | |   |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | |   |     |-<<<NULL>>>
// CHECK:         | | | |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         | | | |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         | | | |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         | | | |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         | | | |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         | | | |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         | | | |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         | | | |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | | |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         | | | |   | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | | |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         | | | |     |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         | | | |-OMPCapturedExprDecl {{.*}} <line:37:23> col:23 implicit used .capture_expr. 'int'
// CHECK:         | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |-OMPCapturedExprDecl {{.*}} <line:38:25> col:25 implicit used .capture_expr. 'int'
// CHECK:         | | | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         | | | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | `-OMPCapturedExprDecl {{.*}} <line:37:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK:         | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK:         | | |     |-BinaryOperator {{.*}} <col:3, line:38:28> 'long' '*'
// CHECK:         | | |     | |-ImplicitCastExpr {{.*}} <line:37:3, col:26> 'long' <IntegralCast>
// CHECK:         | | |     | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK:         | | |     | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK:         | | |     | |   | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK:         | | |     | |   |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         | | |     | |   |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK:         | | |     | |   |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK:         | | |     | |   |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK:         | | |     | |   |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK:         | | |     | |   |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         | | |     | |   |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK:         | | |     | |   |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK:         | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK:         | | |     | `-ImplicitCastExpr {{.*}} <line:38:5, col:28> 'long' <IntegralCast>
// CHECK:         | | |     |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK:         | | |     |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK:         | | |     |     | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
// CHECK:         | | |     |     |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         | | |     |     |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK:         | | |     |     |   `-ParenExpr {{.*}} <col:5> 'int'
// CHECK:         | | |     |     |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
// CHECK:         | | |     |     |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
// CHECK:         | | |     |     |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         | | |     |     |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK:         | | |     |     |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK:         | | |     |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK:         | | |     `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK:         | | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK:         | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | |-attrDetails: AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK:         | |-ImplicitParamDecl {{.*}} <line:35:1> col:1 implicit .global_tid. 'const int'
// CHECK:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK:         | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK:         | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int'
// CHECK:         | | | `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK:         | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int'
// CHECK:         | | | `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK:         | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int'
// CHECK:         | |   `-attrDetails: OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit {{.*}}
// CHECK:         | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:36:1, col:59>
// CHECK:         |   | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK:         |   | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK:         |   | | |-value: Int 2
// CHECK:         |   | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK:         |   | `-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |   | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |   | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   |   | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   |   | | | | |-<<<NULL>>>
// CHECK:         |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         |   |   | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   |   | | | |   |-<<<NULL>>>
// CHECK:         |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         |   |   | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         |   |   | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   |   | | | |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   |   | | | |     |-<<<NULL>>>
// CHECK:         |   |   | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         |   |   | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         |   |   | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         |   |   | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         |   |   | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         |   |   | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   |   | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   |   | | | | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   |   | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   |   | | |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   |   | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK:         |   |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         |   |   | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK:         |   |   | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK:         |   |   | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK:         |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |   |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   |   |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   |   |   | |-<<<NULL>>>
// CHECK:         |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         |   |   |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   |   |   |   |-<<<NULL>>>
// CHECK:         |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         |   |   |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         |   |   |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   |   |   |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   |   |   |     |-<<<NULL>>>
// CHECK:         |   |   |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         |   |   |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         |   |   |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         |   |   |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         |   |   |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         |   |   |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   |   |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   |   |   | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   |   |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   |   |     |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   |   |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |-ImplicitParamDecl {{.*}} <line:35:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK:         |   |-RecordDecl {{.*}} <line:36:1> col:1 implicit struct definition
// CHECK:         |   | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         |   | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK:         |   | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK:         |   | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK:         |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   | | | | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   | | | | |-<<<NULL>>>
// CHECK:         |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         |   | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   | | | |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   | | | |   |-<<<NULL>>>
// CHECK:         |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         |   | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         |   | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   | | | |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   | | | |     |-<<<NULL>>>
// CHECK:         |   | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         |   | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         |   | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         |   | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         |   | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         |   | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   | | | | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   | | | | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   | | |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK:         |   | | |-attrDetails: CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK:         |   | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK:         |   | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK:         |   | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK:         |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK:         |   |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK:         |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   |   | |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   |   | |-<<<NULL>>>
// CHECK:         |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK:         |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK:         |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK:         |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK:         |   |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK:         |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   |   |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   |   |   |-<<<NULL>>>
// CHECK:         |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK:         |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK:         |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK:         |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |   `-ForStmt {{.*}} <line:39:7, line:40:9>
// CHECK:         |   |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK:         |   |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   |   |     |   |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   |   |     |-<<<NULL>>>
// CHECK:         |   |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK:         |   |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK:         |   |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK:         |   |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK:         |   |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK:         |   |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK:         |   |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK:         |   |   | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |   |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK:         |   |   | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |   |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK:         |   |     |-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK:         |   |-OMPCapturedExprDecl {{.*}} <line:37:23> col:23 implicit used .capture_expr. 'int'
// CHECK:         |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |-OMPCapturedExprDecl {{.*}} <line:38:25> col:25 implicit used .capture_expr. 'int'
// CHECK:         |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   `-OMPCapturedExprDecl {{.*}} <line:37:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK:         |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK:         |       |-BinaryOperator {{.*}} <col:3, line:38:28> 'long' '*'
// CHECK:         |       | |-ImplicitCastExpr {{.*}} <line:37:3, col:26> 'long' <IntegralCast>
// CHECK:         |       | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK:         |       | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK:         |       | |   | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
// CHECK:         |       | |   |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK:         |       | |   |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK:         |       | |   |   `-ParenExpr {{.*}} <col:3> 'int'
// CHECK:         |       | |   |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
// CHECK:         |       | |   |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
// CHECK:         |       | |   |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK:         |       | |   |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK:         |       | |   |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK:         |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK:         |       | `-ImplicitCastExpr {{.*}} <line:38:5, col:28> 'long' <IntegralCast>
// CHECK:         |       |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK:         |       |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK:         |       |     | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
// CHECK:         |       |     |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK:         |       |     |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK:         |       |     |   `-ParenExpr {{.*}} <col:5> 'int'
// CHECK:         |       |     |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
// CHECK:         |       |     |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
// CHECK:         |       |     |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK:         |       |     |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK:         |       |     |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK:         |       |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK:         |       `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK:         |         `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK:         |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK:         |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK:         `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
