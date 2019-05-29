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
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:22, line:8:1>
// CHECK-NEXT: |   `-OMPTargetDirective {{.*}} <line:4:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} <line:6:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:5:1, col:47>
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt {{.*}} <col:1, col:47>
// CHECK-NEXT: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:47> openmp_structured_block
// CHECK-NEXT: |       | | | | `-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl {{.*}} <line:5:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt {{.*}} <col:3, line:7:5>
// CHECK-NEXT: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK-NEXT: |       | | | |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl {{.*}} <col:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       | | |     | | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       | | |     | |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     | |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       | | |     | |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:5:1, col:47> openmp_structured_block
// CHECK-NEXT: |       |   | `-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl {{.*}} <line:5:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt {{.*}} <col:3, line:7:5>
// CHECK-NEXT: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt {{.*}} <line:6:3, line:7:5>
// CHECK-NEXT: |       |   | | | | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   | | | `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | `-FieldDecl {{.*}} <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt {{.*}} <col:3, line:7:5>
// CHECK-NEXT: |       |   |   | |-DeclStmt {{.*}} <line:6:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | `-NullStmt {{.*}} <line:7:5> openmp_structured_block
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   `-VarDecl {{.*}} <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl {{.*}} <col:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       |       | | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       |       | |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       | |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       |       | |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |       | |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:10:1, line:16:1> line:10:6 test_two 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:22, col:26> col:26 used y 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:29, line:16:1>
// CHECK-NEXT: |   `-OMPTargetDirective {{.*}} <line:11:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:12:1, col:47>
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt {{.*}} <col:1, col:47>
// CHECK-NEXT: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:47> openmp_structured_block
// CHECK-NEXT: |       | | | | `-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl {{.*}} <line:11:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl {{.*}} <line:12:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       | | | | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl {{.*}} <line:13:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       | | |     | | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       | | |     | |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     | |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       | | |     | |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <line:11:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:12:1, col:47> openmp_structured_block
// CHECK-NEXT: |       |   | `-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       |   |   | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       |   |   |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl {{.*}} <line:11:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl {{.*}} <line:12:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   | | | | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       |   | | | |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   | | | |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr {{.*}} <line:13:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl {{.*}} <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | | `-FieldDecl {{.*}} <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt {{.*}} <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   | |-DeclStmt {{.*}} <line:13:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | `-ForStmt {{.*}} <line:14:5, line:15:7> openmp_structured_block
// CHECK-NEXT: |       |   |   |   |-DeclStmt {{.*}} <line:14:10, col:19>
// CHECK-NEXT: |       |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   `-NullStmt {{.*}} <line:15:7>
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   |-VarDecl {{.*}} <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-VarDecl {{.*}} <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl {{.*}} <line:13:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       |       | | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       |       | |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       | |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       |       | |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |       | |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} <line:14:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:18:1, line:24:1> line:18:6 test_three 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:17, col:21> col:21 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:24, col:28> col:28 used y 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:31, line:24:1>
// CHECK-NEXT: |   `-OMPTargetDirective {{.*}} <line:19:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:20:1, col:59>
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt {{.*}} <col:1, col:59>
// CHECK-NEXT: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:59> openmp_structured_block
// CHECK-NEXT: |       | | | | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK-NEXT: |       | | | | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK-NEXT: |       | | | | |   `-IntegerLiteral {{.*}} <col:57> 'int' 1
// CHECK-NEXT: |       | | | | `-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl {{.*}} <line:19:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl {{.*}} <line:20:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       | | | | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl {{.*}} <line:21:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       | | |     | | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       | | |     | |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     | |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       | | |     | |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <line:19:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:20:1, col:59> openmp_structured_block
// CHECK-NEXT: |       |   | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK-NEXT: |       |   | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK-NEXT: |       |   | |   `-IntegerLiteral {{.*}} <col:57> 'int' 1
// CHECK-NEXT: |       |   | `-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       |   |   | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       |   |   |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl {{.*}} <line:19:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl {{.*}} <line:20:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   | | | | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       |   | | | |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   | | | |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr {{.*}} <line:21:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl {{.*}} <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | | `-FieldDecl {{.*}} <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt {{.*}} <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   | |-DeclStmt {{.*}} <line:21:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | `-ForStmt {{.*}} <line:22:5, line:23:7> openmp_structured_block
// CHECK-NEXT: |       |   |   |   |-DeclStmt {{.*}} <line:22:10, col:19>
// CHECK-NEXT: |       |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   `-NullStmt {{.*}} <line:23:7>
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   |-VarDecl {{.*}} <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-VarDecl {{.*}} <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl {{.*}} <line:21:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl {{.*}} <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       |       | | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       |       | |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       | |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       |       | |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |       | |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} <line:22:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:26:1, line:32:1> line:26:6 test_four 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:30, line:32:1>
// CHECK-NEXT: |   `-OMPTargetDirective {{.*}} <line:27:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:28:1, col:59>
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt {{.*}} <col:1, col:59>
// CHECK-NEXT: |       | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:59> openmp_structured_block
// CHECK-NEXT: |       | | | | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK-NEXT: |       | | | | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK-NEXT: |       | | | | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK-NEXT: |       | | | | `-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | |   | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | |   |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl {{.*}} <line:27:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl {{.*}} <line:28:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | |   |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl {{.*}} <line:29:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl {{.*}} <line:30:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl {{.*}} <line:29:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT: |       | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator {{.*}} <col:3, line:30:28> 'long' '*'
// CHECK-NEXT: |       | | |     | |-ImplicitCastExpr {{.*}} <line:29:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT: |       | | |     | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       | | |     | |   | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       | | |     | |   |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |   |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | |   |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | `-ImplicitCastExpr {{.*}} <line:30:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT: |       | | |     |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK-NEXT: |       | | |     |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK-NEXT: |       | | |     |     | `-BinaryOperator {{.*}} <col:25, col:28> 'int' '+'
// CHECK-NEXT: |       | | |     |     |   |-BinaryOperator {{.*}} <col:25, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |     |   | |-BinaryOperator {{.*}} <col:25, col:18> 'int' '-'
// CHECK-NEXT: |       | | |     |     |   | | |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |     |   | | | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     |     |   | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | | |     |     |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     |     |   `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT: |       | | |     |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT: |       | | |     `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT: |       | | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <line:27:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT: |       | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:28:1, col:59> openmp_structured_block
// CHECK-NEXT: |       |   | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK-NEXT: |       |   | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK-NEXT: |       |   | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK-NEXT: |       |   | `-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       |   |   | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       |   |   |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl {{.*}} <line:27:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl {{.*}} <line:28:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   | | | | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       |   | | | |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   | | | |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   | | | |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   | | `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl {{.*}} <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | | `-FieldDecl {{.*}} <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt {{.*}} <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   | |-DeclStmt {{.*}} <line:29:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   | `-ForStmt {{.*}} <line:30:5, line:31:7>
// CHECK-NEXT: |       |   |   |   |-DeclStmt {{.*}} <line:30:10, col:19>
// CHECK-NEXT: |       |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       |   |   |   `-NullStmt {{.*}} <line:31:7> openmp_structured_block
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   |-VarDecl {{.*}} <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-VarDecl {{.*}} <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl {{.*}} <line:29:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl {{.*}} <line:30:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl {{.*}} <line:29:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT: |       |       |-BinaryOperator {{.*}} <col:3, line:30:28> 'long' '*'
// CHECK-NEXT: |       |       | |-ImplicitCastExpr {{.*}} <line:29:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT: |       |       | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT: |       |       | |   | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT: |       |       | |   |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       | |   |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT: |       |       | |   |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       |       | |   |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | |   |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT: |       |       | `-ImplicitCastExpr {{.*}} <line:30:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT: |       |       |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK-NEXT: |       |       |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK-NEXT: |       |       |     | `-BinaryOperator {{.*}} <col:25, col:28> 'int' '+'
// CHECK-NEXT: |       |       |     |   |-BinaryOperator {{.*}} <col:25, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |     |   | |-BinaryOperator {{.*}} <col:25, col:18> 'int' '-'
// CHECK-NEXT: |       |       |     |   | | |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |     |   | | | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT: |       |       |     |   | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |       |     |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       |     |   `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT: |       |       |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT: |       |       `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT: |       |         `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} <line:29:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} <line:30:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: `-FunctionDecl {{.*}} <line:34:1, line:41:1> line:34:6 test_five 'void (int, int, int)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:30, col:34> col:34 used z 'int'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:37, line:41:1>
// CHECK-NEXT:     `-OMPTargetDirective {{.*}} <line:35:1, col:19>
// CHECK-NEXT:       |-OMPFirstprivateClause {{.*}} <<invalid sloc>> <implicit>
// CHECK-NEXT:       | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:       | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:36:1, col:59>
// CHECK-NEXT:         |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | |-CapturedStmt {{.*}} <col:1, col:59>
// CHECK-NEXT:         | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <col:1, col:59> openmp_structured_block
// CHECK-NEXT:         | | | | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK-NEXT:         | | | | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK-NEXT:         | | | | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK-NEXT:         | | | | `-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         | | | |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         | | | |   | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         | | | |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         | | | |   | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         | | | |   | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | |   | | | |     |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | |   | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | |   | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | |   | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | |   | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | |   | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | |   | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | |   | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | |   | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | |   | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         | | | |   | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         | | | |   | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         | | | |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         | | | |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         | | | |   |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         | | | |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         | | | |   |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         | | | |   |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | |   |   |     |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | |   |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | |   |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | |   |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | |   |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   |     `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | |   |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | |   |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | |   `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | |-ImplicitParamDecl {{.*}} <line:35:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK-NEXT:         | | | |-RecordDecl {{.*}} <line:36:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         | | | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         | | | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         | | | | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | | | | | |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         | | | | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         | | | | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         | | | | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         | | | | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | | | |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | | | | |     |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | | |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         | | | | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         | | | | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         | | | | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         | | | |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | |   | |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         | | | |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         | | | |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | |   |   |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         | | | |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         | | | |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | |   |     |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | | | |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | | |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | | |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |     `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | | | |-OMPCapturedExprDecl {{.*}} <line:37:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT:         | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | | |-OMPCapturedExprDecl {{.*}} <line:38:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT:         | | | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | | `-OMPCapturedExprDecl {{.*}} <line:37:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT:         | | |   `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT:         | | |     |-BinaryOperator {{.*}} <col:3, line:38:28> 'long' '*'
// CHECK-NEXT:         | | |     | |-ImplicitCastExpr {{.*}} <line:37:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT:         | | |     | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT:         | | |     | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT:         | | |     | |   | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT:         | | |     | |   |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT:         | | |     | |   |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT:         | | |     | |   |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     | |   |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT:         | | |     | |   |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | |     | |   |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:         | | |     | |   |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT:         | | |     | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT:         | | |     | `-ImplicitCastExpr {{.*}} <line:38:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT:         | | |     |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK-NEXT:         | | |     |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK-NEXT:         | | |     |     | `-BinaryOperator {{.*}} <col:25, col:28> 'int' '+'
// CHECK-NEXT:         | | |     |     |   |-BinaryOperator {{.*}} <col:25, <invalid sloc>> 'int' '-'
// CHECK-NEXT:         | | |     |     |   | |-BinaryOperator {{.*}} <col:25, col:18> 'int' '-'
// CHECK-NEXT:         | | |     |     |   | | |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     |     |   | | | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT:         | | |     |     |   | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | | |     |     |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:         | | |     |     |   `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT:         | | |     |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT:         | | |     `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT:         | | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:         | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         | |-AlwaysInlineAttr {{.*}} <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <line:35:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK-NEXT:         | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT:         | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT:         | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int'
// CHECK-NEXT:         | |   `-OMPCaptureKindAttr {{.*}} <<invalid sloc>> Implicit 9
// CHECK-NEXT:         | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |-OMPTeamsDistributeParallelForSimdDirective {{.*}} <line:36:1, col:59> openmp_structured_block
// CHECK-NEXT:         |   | |-OMPCollapseClause {{.*}} <col:48, col:58>
// CHECK-NEXT:         |   | | `-ConstantExpr {{.*}} <col:57> 'int'
// CHECK-NEXT:         |   | |   `-IntegerLiteral {{.*}} <col:57> 'int' 2
// CHECK-NEXT:         |   | `-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         |   |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT:         |   |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         |   |   | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         |   |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT:         |   |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         |   |   | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         |   |   | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   |   | | | |     |-<<<NULL>>>
// CHECK-NEXT:         |   |   | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   |   | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   |   | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   |   | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   |   | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   | | |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   |   | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   |   | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   |   | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   |   | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |   |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         |   |   | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         |   |   | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         |   |   | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         |   |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         |   |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   |   |   | |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         |   |   |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         |   |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         |   |   |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         |   |   |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   |   |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   |   |   |     |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   |   |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   |   |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   |   |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   |   | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   |   |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   |     `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   |   |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   |   |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   |   `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   |-ImplicitParamDecl {{.*}} <line:35:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK-NEXT:         |   |-RecordDecl {{.*}} <line:36:1> col:1 implicit struct definition
// CHECK-NEXT:         |   | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         |   | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         |   | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         |   | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         |   |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | |-CapturedStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   | | |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | | | |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   | | | | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         |   | | | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | | | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   | | | | |-<<<NULL>>>
// CHECK-NEXT:         |   | | | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   | | | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   | | | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   | | | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   | | | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   | | | | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         |   | | | |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         |   | | | |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | | | |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   | | | |   |-<<<NULL>>>
// CHECK-NEXT:         |   | | | |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   | | | |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   | | | |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   | | | |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   | | | |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   | | | |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         |   | | | |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         |   | | | |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | | | |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   | | | |     |-<<<NULL>>>
// CHECK-NEXT:         |   | | | |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   | | | |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   | | | |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   | | | |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   | | | |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   | | | |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   | | | |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | | | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   | | | |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | | | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   | | | `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | | |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   | | |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   | | |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   | | `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   | |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   | |-RecordDecl {{.*}} <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |   | | |-CapturedRecordAttr {{.*}} <<invalid sloc>> Implicit
// CHECK-NEXT:         |   | | |-FieldDecl {{.*}} <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         |   | | |-FieldDecl {{.*}} <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         |   | | `-FieldDecl {{.*}} <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         |   | `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   |-ForStmt {{.*}} <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   | |-DeclStmt {{.*}} <line:37:8, col:17>
// CHECK-NEXT:         |   |   | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   |   | |-<<<NULL>>>
// CHECK-NEXT:         |   |   | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   |   | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   |   | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   | `-ForStmt {{.*}} <line:38:5, line:40:9>
// CHECK-NEXT:         |   |   |   |-DeclStmt {{.*}} <line:38:10, col:19>
// CHECK-NEXT:         |   |   |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   |   |   |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   |   |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   |   |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |   `-ForStmt {{.*}} <line:39:7, line:40:9> openmp_structured_block
// CHECK-NEXT:         |   |   |     |-DeclStmt {{.*}} <line:39:12, col:21>
// CHECK-NEXT:         |   |   |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   |   |     |-<<<NULL>>>
// CHECK-NEXT:         |   |   |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   |   |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
// CHECK-NEXT:         |   |   |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   |   |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         |   |   |     `-NullStmt {{.*}} <line:40:9>
// CHECK-NEXT:         |   |   |-ImplicitParamDecl {{.*}} <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.lb. 'const unsigned long'
// CHECK-NEXT:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit used .previous.ub. 'const unsigned long'
// CHECK-NEXT:         |   |   |-ImplicitParamDecl {{.*}} <col:1> col:1 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   |-VarDecl {{.*}} <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |   |   |-VarDecl {{.*}} <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |   |   `-VarDecl {{.*}} <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |     `-IntegerLiteral {{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |   |-OMPCapturedExprDecl {{.*}} <line:37:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT:         |   | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |   |-OMPCapturedExprDecl {{.*}} <line:38:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT:         |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         |   `-OMPCapturedExprDecl {{.*}} <line:37:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT:         |     `-BinaryOperator {{.*}} <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT:         |       |-BinaryOperator {{.*}} <col:3, line:38:28> 'long' '*'
// CHECK-NEXT:         |       | |-ImplicitCastExpr {{.*}} <line:37:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT:         |       | | `-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
// CHECK-NEXT:         |       | |   |-ParenExpr {{.*}} <col:3> 'int'
// CHECK-NEXT:         |       | |   | `-BinaryOperator {{.*}} <col:23, col:26> 'int' '+'
// CHECK-NEXT:         |       | |   |   |-BinaryOperator {{.*}} <col:23, <invalid sloc>> 'int' '-'
// CHECK-NEXT:         |       | |   |   | |-BinaryOperator {{.*}} <col:23, col:16> 'int' '-'
// CHECK-NEXT:         |       | |   |   | | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |       | |   |   | | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT:         |       | |   |   | | `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         |       | |   |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:         |       | |   |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT:         |       | |   `-IntegerLiteral {{.*}} <col:26> 'int' 1
// CHECK-NEXT:         |       | `-ImplicitCastExpr {{.*}} <line:38:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT:         |       |   `-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
// CHECK-NEXT:         |       |     |-ParenExpr {{.*}} <col:5> 'int'
// CHECK-NEXT:         |       |     | `-BinaryOperator {{.*}} <col:25, col:28> 'int' '+'
// CHECK-NEXT:         |       |     |   |-BinaryOperator {{.*}} <col:25, <invalid sloc>> 'int' '-'
// CHECK-NEXT:         |       |     |   | |-BinaryOperator {{.*}} <col:25, col:18> 'int' '-'
// CHECK-NEXT:         |       |     |   | | |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |       |     |   | | | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue OMPCapturedExpr {{.*}} '.capture_expr.' 'int'
// CHECK-NEXT:         |       |     |   | | `-IntegerLiteral {{.*}} <col:18> 'int' 0
// CHECK-NEXT:         |       |     |   | `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:         |       |     |   `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT:         |       |     `-IntegerLiteral {{.*}} <col:28> 'int' 1
// CHECK-NEXT:         |       `-ImplicitCastExpr {{.*}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT:         |         `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
// CHECK-NEXT:         |-DeclRefExpr {{.*}} <line:37:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |-DeclRefExpr {{.*}} <line:38:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         `-DeclRefExpr {{.*}} <line:39:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
