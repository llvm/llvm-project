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

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:8:1> line:3:6 test_one 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_1:0x[a-z0-9]*]] <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_2:0x[a-z0-9]*]] <col:22, line:8:1>
// CHECK-NEXT: |   `-OMPTargetDirective [[ADDR_3:0x[a-z0-9]*]] <line:4:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:6:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_6:0x[a-z0-9]*]] <line:5:1, col:47>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_7:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_8:0x[a-z0-9]*]] <col:1, col:47>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_9:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_10:0x[a-z0-9]*]] <col:1, col:47>
// CHECK-NEXT: |       | | | | `-CapturedStmt [[ADDR_11:0x[a-z0-9]*]] <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | |   |-CapturedDecl [[ADDR_12:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt [[ADDR_13:0x[a-z0-9]*]] <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl [[ADDR_14:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt [[ADDR_15:0x[a-z0-9]*]] <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt [[ADDR_16:0x[a-z0-9]*]] <line:6:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl [[ADDR_17:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral [[ADDR_18:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator [[ADDR_19:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr [[ADDR_20:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr [[ADDR_21:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr [[ADDR_22:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr [[ADDR_23:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator [[ADDR_24:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr [[ADDR_25:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-NullStmt [[ADDR_26:0x[a-z0-9]*]] <line:7:5>
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_27:0x[a-z0-9]*]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_28:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_29:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_30:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_32:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_33:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_34:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_35:0x[a-z0-9]*]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_36:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_37:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl [[ADDR_38:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr [[ADDR_39:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_40:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_41:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl [[ADDR_42:0x[a-z0-9]*]] <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt [[ADDR_15]] <col:3, line:7:5>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_43:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_44:0x[a-z0-9]*]] <line:4:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_45:0x[a-z0-9]*]] <line:5:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_46:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_47:0x[a-z0-9]*]] <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl [[ADDR_12]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt [[ADDR_13]] <col:3, line:7:5>
// CHECK-NEXT: |       | | | | | |-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt [[ADDR_15]] <line:6:3, line:7:5>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | | | | `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_30]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_32]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_33]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr [[ADDR_34]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_35]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_36]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_37]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl [[ADDR_38]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr [[ADDR_39]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_40]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_41]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | `-FieldDecl [[ADDR_42]] <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt [[ADDR_15]] <col:3, line:7:5>
// CHECK-NEXT: |       | | | |   | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       | | | |   `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl [[ADDR_48:0x[a-z0-9]*]] <col:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl [[ADDR_49:0x[a-z0-9]*]] <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | |   `-BinaryOperator [[ADDR_50:0x[a-z0-9]*]] <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator [[ADDR_51:0x[a-z0-9]*]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |-ParenExpr [[ADDR_52:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | | `-BinaryOperator [[ADDR_53:0x[a-z0-9]*]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |-ImplicitCastExpr [[ADDR_54:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   | `-DeclRefExpr [[ADDR_55:0x[a-z0-9]*]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_48]] '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   `-ParenExpr [[ADDR_56:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | |     `-BinaryOperator [[ADDR_57:0x[a-z0-9]*]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       | | |     | |       |-BinaryOperator [[ADDR_58:0x[a-z0-9]*]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       | | |     | |       | |-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |       | `-IntegerLiteral [[ADDR_59:0x[a-z0-9]*]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | |       `-IntegerLiteral [[ADDR_60:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | `-IntegerLiteral [[ADDR_59]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     `-IntegerLiteral [[ADDR_61:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_62:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_63:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_64:0x[a-z0-9]*]] <line:4:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_65:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_66:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_67:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_68:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_69:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_70:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_71:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_72:0x[a-z0-9]*]] <line:6:23> col:23 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_73:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_9]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_10]] <line:5:1, col:47>
// CHECK-NEXT: |       |   | `-CapturedStmt [[ADDR_11]] <line:6:3, line:7:5>
// CHECK-NEXT: |       |   |   |-CapturedDecl [[ADDR_12]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt [[ADDR_13]] <line:6:3, line:7:5>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt [[ADDR_15]] <line:6:3, line:7:5>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_30]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_32]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_33]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_34]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_35]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_36]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_37]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl [[ADDR_38]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr [[ADDR_39]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_40]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_41]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl [[ADDR_42]] <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt [[ADDR_15]] <col:3, line:7:5>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   |   `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_43]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_44]] <line:4:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:4:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_45]] <line:5:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_46]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_47]] <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl [[ADDR_12]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt [[ADDR_13]] <col:3, line:7:5>
// CHECK-NEXT: |       |   | | |-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt [[ADDR_15]] <line:6:3, line:7:5>
// CHECK-NEXT: |       |   | | | | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   | | | `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_30]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_32]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_33]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | `-DeclRefExpr [[ADDR_34]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_35]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_36]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_37]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl [[ADDR_38]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr [[ADDR_39]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_40]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_41]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | `-FieldDecl [[ADDR_42]] <line:6:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl [[ADDR_14]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt [[ADDR_15]] <col:3, line:7:5>
// CHECK-NEXT: |       |   |   | |-DeclStmt [[ADDR_16]] <line:6:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl [[ADDR_17]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator [[ADDR_19]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr [[ADDR_20]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr [[ADDR_21]] <col:19> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator [[ADDR_24]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_25]] <col:26> 'int' {{.*}}Var [[ADDR_17]] 'i' 'int'
// CHECK-NEXT: |       |   |   | `-NullStmt [[ADDR_26]] <line:7:5>
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_27]] <line:5:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_28]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_29]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:5:1) *const restrict'
// CHECK-NEXT: |       |   |   `-VarDecl [[ADDR_17]] <line:6:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl [[ADDR_48]] <col:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr [[ADDR_22]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_23]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl [[ADDR_49]] <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |     `-BinaryOperator [[ADDR_50]] <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |-BinaryOperator [[ADDR_51]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |-ParenExpr [[ADDR_52]] <col:3> 'int'
// CHECK-NEXT: |       |       | | `-BinaryOperator [[ADDR_53]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       |       | |   |-ImplicitCastExpr [[ADDR_54]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   | `-DeclRefExpr [[ADDR_55]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_48]] '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   `-ParenExpr [[ADDR_56]] <col:3> 'int'
// CHECK-NEXT: |       |       | |     `-BinaryOperator [[ADDR_57]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       |       | |       |-BinaryOperator [[ADDR_58]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       |       | |       | |-IntegerLiteral [[ADDR_18]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |       | `-IntegerLiteral [[ADDR_59]] <col:26> 'int' 1
// CHECK-NEXT: |       |       | |       `-IntegerLiteral [[ADDR_60]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | `-IntegerLiteral [[ADDR_59]] <col:26> 'int' 1
// CHECK-NEXT: |       |       `-IntegerLiteral [[ADDR_61]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_74:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_75:0x[a-z0-9]*]] <line:10:1, line:16:1> line:10:6 test_two 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_76:0x[a-z0-9]*]] <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_77:0x[a-z0-9]*]] <col:22, col:26> col:26 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_78:0x[a-z0-9]*]] <col:29, line:16:1>
// CHECK-NEXT: |   `-OMPTargetDirective [[ADDR_79:0x[a-z0-9]*]] <line:11:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_80:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr [[ADDR_81:0x[a-z0-9]*]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_82:0x[a-z0-9]*]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_83:0x[a-z0-9]*]] <line:12:1, col:47>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_84:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_85:0x[a-z0-9]*]] <col:1, col:47>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_86:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_87:0x[a-z0-9]*]] <col:1, col:47>
// CHECK-NEXT: |       | | | | `-CapturedStmt [[ADDR_88:0x[a-z0-9]*]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   |-CapturedDecl [[ADDR_89:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt [[ADDR_90:0x[a-z0-9]*]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl [[ADDR_91:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt [[ADDR_92:0x[a-z0-9]*]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt [[ADDR_93:0x[a-z0-9]*]] <line:13:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl [[ADDR_94:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral [[ADDR_95:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator [[ADDR_96:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr [[ADDR_97:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr [[ADDR_98:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr [[ADDR_99:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr [[ADDR_100:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator [[ADDR_101:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr [[ADDR_102:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-ForStmt [[ADDR_103:0x[a-z0-9]*]] <line:14:5, line:15:7>
// CHECK-NEXT: |       | | | |   | | | |   |-DeclStmt [[ADDR_104:0x[a-z0-9]*]] <line:14:10, col:19>
// CHECK-NEXT: |       | | | |   | | | |   | `-VarDecl [[ADDR_105:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | |   |   `-IntegerLiteral [[ADDR_106:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | |   |-BinaryOperator [[ADDR_107:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | |   | |-ImplicitCastExpr [[ADDR_108:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   | | `-DeclRefExpr [[ADDR_109:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   | `-ImplicitCastExpr [[ADDR_110:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   |   `-DeclRefExpr [[ADDR_111:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | |   | | | |   |-UnaryOperator [[ADDR_112:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | |   | `-DeclRefExpr [[ADDR_113:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   `-NullStmt [[ADDR_114:0x[a-z0-9]*]] <line:15:7>
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_115:0x[a-z0-9]*]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_116:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_117:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_118:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_119:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_120:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_121:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_122:0x[a-z0-9]*]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_123:0x[a-z0-9]*]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_124:0x[a-z0-9]*]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_125:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_126:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl [[ADDR_127:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr [[ADDR_128:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_129:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_130:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_131:0x[a-z0-9]*]] <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl [[ADDR_132:0x[a-z0-9]*]] <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       | | | |   |   |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       | | | |   |   |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |-DeclRefExpr [[ADDR_133:0x[a-z0-9]*]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_134:0x[a-z0-9]*]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_135:0x[a-z0-9]*]] <line:11:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_136:0x[a-z0-9]*]] <line:12:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_137:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl [[ADDR_138:0x[a-z0-9]*]] <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_139:0x[a-z0-9]*]] <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl [[ADDR_89]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt [[ADDR_90]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | | | |-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       | | | | | | |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       | | | | | | |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | | | | |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | | | | |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_118]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_119]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_120]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_121]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_122]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr [[ADDR_123]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_124]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_125]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_126]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl [[ADDR_127]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr [[ADDR_128]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_129]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_130]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_131]] <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | | `-FieldDecl [[ADDR_132]] <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       | | | |   | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       | | | |   |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       | | | |   |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       | | | |   |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl [[ADDR_140:0x[a-z0-9]*]] <line:13:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl [[ADDR_141:0x[a-z0-9]*]] <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | |   `-BinaryOperator [[ADDR_142:0x[a-z0-9]*]] <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator [[ADDR_143:0x[a-z0-9]*]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |-ParenExpr [[ADDR_144:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | | `-BinaryOperator [[ADDR_145:0x[a-z0-9]*]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |-ImplicitCastExpr [[ADDR_146:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   | `-DeclRefExpr [[ADDR_147:0x[a-z0-9]*]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_140]] '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   `-ParenExpr [[ADDR_148:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | |     `-BinaryOperator [[ADDR_149:0x[a-z0-9]*]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       | | |     | |       |-BinaryOperator [[ADDR_150:0x[a-z0-9]*]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       | | |     | |       | |-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |       | `-IntegerLiteral [[ADDR_151:0x[a-z0-9]*]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | |       `-IntegerLiteral [[ADDR_152:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | `-IntegerLiteral [[ADDR_151]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     `-IntegerLiteral [[ADDR_153:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |-DeclRefExpr [[ADDR_154:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_155:0x[a-z0-9]*]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_156:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_157:0x[a-z0-9]*]] <line:11:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_158:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_159:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_160:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_161:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_162:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_163:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_164:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl [[ADDR_165:0x[a-z0-9]*]] <line:13:23> col:23 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr [[ADDR_166:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_167:0x[a-z0-9]*]] <line:14:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_168:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_86]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_87]] <line:12:1, col:47>
// CHECK-NEXT: |       |   | `-CapturedStmt [[ADDR_88]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   |-CapturedDecl [[ADDR_89]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt [[ADDR_90]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       |   |   | | | |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       |   |   | | | |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   | | | |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   |   | | | |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_118]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_119]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_120]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_121]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_122]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_123]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_124]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_125]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_126]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl [[ADDR_127]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr [[ADDR_128]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_129]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_130]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_131]] <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl [[ADDR_132]] <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       |   |   |   |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       |   |   |   |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   |   |   |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   |   |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |-DeclRefExpr [[ADDR_133]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_134]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_135]] <line:11:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:11:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_136]] <line:12:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_137]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl [[ADDR_138]] <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_139]] <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl [[ADDR_89]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt [[ADDR_90]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   | | |-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   | | | | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       |   | | | |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       |   | | | |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   | | | |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   | | | |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   | | | |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   | | | |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_118]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_119]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_120]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_121]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_122]] <line:13:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   | | `-DeclRefExpr [[ADDR_123]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_124]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_125]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_126]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl [[ADDR_127]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr [[ADDR_128]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_129]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_130]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_131]] <line:13:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | | `-FieldDecl [[ADDR_132]] <line:14:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl [[ADDR_91]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt [[ADDR_92]] <line:13:3, line:15:7>
// CHECK-NEXT: |       |   |   | |-DeclStmt [[ADDR_93]] <line:13:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl [[ADDR_94]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator [[ADDR_96]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr [[ADDR_97]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr [[ADDR_98]] <col:19> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator [[ADDR_101]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_102]] <col:26> 'int' {{.*}}Var [[ADDR_94]] 'i' 'int'
// CHECK-NEXT: |       |   |   | `-ForStmt [[ADDR_103]] <line:14:5, line:15:7>
// CHECK-NEXT: |       |   |   |   |-DeclStmt [[ADDR_104]] <line:14:10, col:19>
// CHECK-NEXT: |       |   |   |   | `-VarDecl [[ADDR_105]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |-BinaryOperator [[ADDR_107]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   | |-ImplicitCastExpr [[ADDR_108]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_109]] <col:21> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ImplicitCastExpr [[ADDR_110]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   `-DeclRefExpr [[ADDR_111]] <col:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |       |   |   |   |-UnaryOperator [[ADDR_112]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | `-DeclRefExpr [[ADDR_113]] <col:28> 'int' {{.*}}Var [[ADDR_105]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   `-NullStmt [[ADDR_114]] <line:15:7>
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_115]] <line:12:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_116]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_117]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:12:1) *const restrict'
// CHECK-NEXT: |       |   |   |-VarDecl [[ADDR_94]] <line:13:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | `-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-VarDecl [[ADDR_105]] <line:14:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral [[ADDR_106]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl [[ADDR_140]] <line:13:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr [[ADDR_99]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_100]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl [[ADDR_141]] <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |     `-BinaryOperator [[ADDR_142]] <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |-BinaryOperator [[ADDR_143]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |-ParenExpr [[ADDR_144]] <col:3> 'int'
// CHECK-NEXT: |       |       | | `-BinaryOperator [[ADDR_145]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       |       | |   |-ImplicitCastExpr [[ADDR_146]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   | `-DeclRefExpr [[ADDR_147]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_140]] '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   `-ParenExpr [[ADDR_148]] <col:3> 'int'
// CHECK-NEXT: |       |       | |     `-BinaryOperator [[ADDR_149]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       |       | |       |-BinaryOperator [[ADDR_150]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       |       | |       | |-IntegerLiteral [[ADDR_95]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |       | `-IntegerLiteral [[ADDR_151]] <col:26> 'int' 1
// CHECK-NEXT: |       |       | |       `-IntegerLiteral [[ADDR_152]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | `-IntegerLiteral [[ADDR_151]] <col:26> 'int' 1
// CHECK-NEXT: |       |       `-IntegerLiteral [[ADDR_153]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_169:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_76]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_170:0x[a-z0-9]*]] <line:14:25> 'int' {{.*}}ParmVar [[ADDR_77]] 'y' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_171:0x[a-z0-9]*]] <line:18:1, line:24:1> line:18:6 test_three 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_172:0x[a-z0-9]*]] <col:17, col:21> col:21 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_173:0x[a-z0-9]*]] <col:24, col:28> col:28 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_174:0x[a-z0-9]*]] <col:31, line:24:1>
// CHECK-NEXT: |   `-OMPTargetDirective [[ADDR_175:0x[a-z0-9]*]] <line:19:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_176:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr [[ADDR_177:0x[a-z0-9]*]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_178:0x[a-z0-9]*]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_179:0x[a-z0-9]*]] <line:20:1, col:59>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_180:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_181:0x[a-z0-9]*]] <col:1, col:59>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_182:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_183:0x[a-z0-9]*]] <col:1, col:59>
// CHECK-NEXT: |       | | | | |-OMPCollapseClause [[ADDR_184:0x[a-z0-9]*]] <col:48, col:58>
// CHECK-NEXT: |       | | | | | `-ConstantExpr [[ADDR_185:0x[a-z0-9]*]] <col:57> 'int'
// CHECK-NEXT: |       | | | | |   |-value: Int 1
// CHECK-NEXT: |       | | | | |   `-IntegerLiteral [[ADDR_186:0x[a-z0-9]*]] <col:57> 'int' 1
// CHECK-NEXT: |       | | | | `-CapturedStmt [[ADDR_187:0x[a-z0-9]*]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   |-CapturedDecl [[ADDR_188:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt [[ADDR_189:0x[a-z0-9]*]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl [[ADDR_190:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt [[ADDR_191:0x[a-z0-9]*]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt [[ADDR_192:0x[a-z0-9]*]] <line:21:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl [[ADDR_193:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral [[ADDR_194:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator [[ADDR_195:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr [[ADDR_196:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr [[ADDR_197:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr [[ADDR_198:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr [[ADDR_199:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator [[ADDR_200:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr [[ADDR_201:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-ForStmt [[ADDR_202:0x[a-z0-9]*]] <line:22:5, line:23:7>
// CHECK-NEXT: |       | | | |   | | | |   |-DeclStmt [[ADDR_203:0x[a-z0-9]*]] <line:22:10, col:19>
// CHECK-NEXT: |       | | | |   | | | |   | `-VarDecl [[ADDR_204:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | |   |   `-IntegerLiteral [[ADDR_205:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | |   |-BinaryOperator [[ADDR_206:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | |   | |-ImplicitCastExpr [[ADDR_207:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   | | `-DeclRefExpr [[ADDR_208:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   | `-ImplicitCastExpr [[ADDR_209:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   |   `-DeclRefExpr [[ADDR_210:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | |   | | | |   |-UnaryOperator [[ADDR_211:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | |   | `-DeclRefExpr [[ADDR_212:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   `-NullStmt [[ADDR_213:0x[a-z0-9]*]] <line:23:7>
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_214:0x[a-z0-9]*]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_215:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_216:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_217:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_218:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_219:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_220:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_221:0x[a-z0-9]*]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_222:0x[a-z0-9]*]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_223:0x[a-z0-9]*]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_224:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_225:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl [[ADDR_226:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr [[ADDR_227:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_228:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_229:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_230:0x[a-z0-9]*]] <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl [[ADDR_231:0x[a-z0-9]*]] <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       | | | |   |   |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       | | | |   |   |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |-DeclRefExpr [[ADDR_232:0x[a-z0-9]*]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_233:0x[a-z0-9]*]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_234:0x[a-z0-9]*]] <line:19:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_235:0x[a-z0-9]*]] <line:20:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_236:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl [[ADDR_237:0x[a-z0-9]*]] <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_238:0x[a-z0-9]*]] <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl [[ADDR_188]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt [[ADDR_189]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | | | |-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       | | | | | | |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       | | | | | | |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | | | | |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | | | | |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_217]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_218]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_219]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_220]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_221]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr [[ADDR_222]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_223]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_224]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_225]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl [[ADDR_226]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr [[ADDR_227]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_228]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_229]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_230]] <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | | `-FieldDecl [[ADDR_231]] <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       | | | |   | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       | | | |   |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       | | | |   |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       | | | |   |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl [[ADDR_239:0x[a-z0-9]*]] <line:21:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl [[ADDR_240:0x[a-z0-9]*]] <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | |   `-BinaryOperator [[ADDR_241:0x[a-z0-9]*]] <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator [[ADDR_242:0x[a-z0-9]*]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |-ParenExpr [[ADDR_243:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | | `-BinaryOperator [[ADDR_244:0x[a-z0-9]*]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |-ImplicitCastExpr [[ADDR_245:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   | `-DeclRefExpr [[ADDR_246:0x[a-z0-9]*]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_239]] '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   `-ParenExpr [[ADDR_247:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | |     `-BinaryOperator [[ADDR_248:0x[a-z0-9]*]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       | | |     | |       |-BinaryOperator [[ADDR_249:0x[a-z0-9]*]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       | | |     | |       | |-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |       | `-IntegerLiteral [[ADDR_250:0x[a-z0-9]*]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | |       `-IntegerLiteral [[ADDR_251:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | `-IntegerLiteral [[ADDR_250]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     `-IntegerLiteral [[ADDR_252:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |-DeclRefExpr [[ADDR_253:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_254:0x[a-z0-9]*]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_255:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_256:0x[a-z0-9]*]] <line:19:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_257:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_258:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_259:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_260:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_261:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_262:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_263:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl [[ADDR_264:0x[a-z0-9]*]] <line:21:23> col:23 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr [[ADDR_265:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_266:0x[a-z0-9]*]] <line:22:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_267:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_182]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_183]] <line:20:1, col:59>
// CHECK-NEXT: |       |   | |-OMPCollapseClause [[ADDR_184]] <col:48, col:58>
// CHECK-NEXT: |       |   | | `-ConstantExpr [[ADDR_185]] <col:57> 'int'
// CHECK-NEXT: |       |   | |   |-value: Int 1
// CHECK-NEXT: |       |   | |   `-IntegerLiteral [[ADDR_186]] <col:57> 'int' 1
// CHECK-NEXT: |       |   | `-CapturedStmt [[ADDR_187]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   |-CapturedDecl [[ADDR_188]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt [[ADDR_189]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       |   |   | | | |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       |   |   | | | |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   | | | |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   |   | | | |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_217]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_218]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_219]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_220]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_221]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_222]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_223]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_224]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_225]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl [[ADDR_226]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr [[ADDR_227]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_228]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_229]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_230]] <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl [[ADDR_231]] <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       |   |   |   |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       |   |   |   |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   |   |   |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   |   |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |-DeclRefExpr [[ADDR_232]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_233]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_234]] <line:19:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:19:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_235]] <line:20:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_236]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl [[ADDR_237]] <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_238]] <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl [[ADDR_188]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt [[ADDR_189]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   | | |-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   | | | | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       |   | | | |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       |   | | | |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   | | | |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   | | | |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   | | | |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   | | | |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_217]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_218]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_219]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_220]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_221]] <line:21:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   | | `-DeclRefExpr [[ADDR_222]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_223]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_224]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_225]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl [[ADDR_226]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr [[ADDR_227]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_228]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_229]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_230]] <line:21:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | | `-FieldDecl [[ADDR_231]] <line:22:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl [[ADDR_190]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt [[ADDR_191]] <line:21:3, line:23:7>
// CHECK-NEXT: |       |   |   | |-DeclStmt [[ADDR_192]] <line:21:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl [[ADDR_193]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator [[ADDR_195]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr [[ADDR_196]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr [[ADDR_197]] <col:19> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator [[ADDR_200]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_201]] <col:26> 'int' {{.*}}Var [[ADDR_193]] 'i' 'int'
// CHECK-NEXT: |       |   |   | `-ForStmt [[ADDR_202]] <line:22:5, line:23:7>
// CHECK-NEXT: |       |   |   |   |-DeclStmt [[ADDR_203]] <line:22:10, col:19>
// CHECK-NEXT: |       |   |   |   | `-VarDecl [[ADDR_204]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |-BinaryOperator [[ADDR_206]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   | |-ImplicitCastExpr [[ADDR_207]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_208]] <col:21> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ImplicitCastExpr [[ADDR_209]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   `-DeclRefExpr [[ADDR_210]] <col:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |       |   |   |   |-UnaryOperator [[ADDR_211]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | `-DeclRefExpr [[ADDR_212]] <col:28> 'int' {{.*}}Var [[ADDR_204]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   `-NullStmt [[ADDR_213]] <line:23:7>
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_214]] <line:20:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_215]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_216]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:20:1) *const restrict'
// CHECK-NEXT: |       |   |   |-VarDecl [[ADDR_193]] <line:21:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | `-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-VarDecl [[ADDR_204]] <line:22:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral [[ADDR_205]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl [[ADDR_239]] <line:21:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr [[ADDR_198]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_199]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl [[ADDR_240]] <col:3, <invalid sloc>> col:3 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |     `-BinaryOperator [[ADDR_241]] <col:3, <invalid sloc>> 'int' '-'
// CHECK-NEXT: |       |       |-BinaryOperator [[ADDR_242]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |-ParenExpr [[ADDR_243]] <col:3> 'int'
// CHECK-NEXT: |       |       | | `-BinaryOperator [[ADDR_244]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       |       | |   |-ImplicitCastExpr [[ADDR_245]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   | `-DeclRefExpr [[ADDR_246]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_239]] '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   `-ParenExpr [[ADDR_247]] <col:3> 'int'
// CHECK-NEXT: |       |       | |     `-BinaryOperator [[ADDR_248]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       |       | |       |-BinaryOperator [[ADDR_249]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       |       | |       | |-IntegerLiteral [[ADDR_194]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |       | `-IntegerLiteral [[ADDR_250]] <col:26> 'int' 1
// CHECK-NEXT: |       |       | |       `-IntegerLiteral [[ADDR_251]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | `-IntegerLiteral [[ADDR_250]] <col:26> 'int' 1
// CHECK-NEXT: |       |       `-IntegerLiteral [[ADDR_252]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_268:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_172]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_269:0x[a-z0-9]*]] <line:22:25> 'int' {{.*}}ParmVar [[ADDR_173]] 'y' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_270:0x[a-z0-9]*]] <line:26:1, line:32:1> line:26:6 test_four 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_271:0x[a-z0-9]*]] <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_272:0x[a-z0-9]*]] <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_273:0x[a-z0-9]*]] <col:30, line:32:1>
// CHECK-NEXT: |   `-OMPTargetDirective [[ADDR_274:0x[a-z0-9]*]] <line:27:1, col:19>
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_275:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr [[ADDR_276:0x[a-z0-9]*]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_277:0x[a-z0-9]*]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_278:0x[a-z0-9]*]] <line:28:1, col:59>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_279:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_280:0x[a-z0-9]*]] <col:1, col:59>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_281:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_282:0x[a-z0-9]*]] <col:1, col:59>
// CHECK-NEXT: |       | | | | |-OMPCollapseClause [[ADDR_283:0x[a-z0-9]*]] <col:48, col:58>
// CHECK-NEXT: |       | | | | | `-ConstantExpr [[ADDR_284:0x[a-z0-9]*]] <col:57> 'int'
// CHECK-NEXT: |       | | | | |   |-value: Int 2
// CHECK-NEXT: |       | | | | |   `-IntegerLiteral [[ADDR_285:0x[a-z0-9]*]] <col:57> 'int' 2
// CHECK-NEXT: |       | | | | `-CapturedStmt [[ADDR_286:0x[a-z0-9]*]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   |-CapturedDecl [[ADDR_287:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | |-CapturedStmt [[ADDR_288:0x[a-z0-9]*]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   | | |-CapturedDecl [[ADDR_289:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   | | | |-ForStmt [[ADDR_290:0x[a-z0-9]*]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   | | | | |-DeclStmt [[ADDR_291:0x[a-z0-9]*]] <line:29:8, col:17>
// CHECK-NEXT: |       | | | |   | | | | | `-VarDecl [[ADDR_292:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | |   `-IntegerLiteral [[ADDR_293:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | | |-BinaryOperator [[ADDR_294:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | | | |-ImplicitCastExpr [[ADDR_295:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | | | `-DeclRefExpr [[ADDR_296:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | | `-ImplicitCastExpr [[ADDR_297:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | | |   `-DeclRefExpr [[ADDR_298:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | | | |-UnaryOperator [[ADDR_299:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | | | `-DeclRefExpr [[ADDR_300:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | | `-ForStmt [[ADDR_301:0x[a-z0-9]*]] <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | |   | | | |   |-DeclStmt [[ADDR_302:0x[a-z0-9]*]] <line:30:10, col:19>
// CHECK-NEXT: |       | | | |   | | | |   | `-VarDecl [[ADDR_303:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | |   |   `-IntegerLiteral [[ADDR_304:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | | | |   |-BinaryOperator [[ADDR_305:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   | | | |   | |-ImplicitCastExpr [[ADDR_306:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   | | `-DeclRefExpr [[ADDR_307:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   | `-ImplicitCastExpr [[ADDR_308:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | |   |   `-DeclRefExpr [[ADDR_309:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | |   | | | |   |-UnaryOperator [[ADDR_310:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | | |   | `-DeclRefExpr [[ADDR_311:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | | |   `-NullStmt [[ADDR_312:0x[a-z0-9]*]] <line:31:7>
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_313:0x[a-z0-9]*]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_314:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-ImplicitParamDecl [[ADDR_315:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   | | | |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | | | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | | | `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | | |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_316:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_317:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_318:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_319:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | |   | | |-DeclRefExpr [[ADDR_320:0x[a-z0-9]*]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_321:0x[a-z0-9]*]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_322:0x[a-z0-9]*]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_323:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   | |-ImplicitParamDecl [[ADDR_324:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   | |-RecordDecl [[ADDR_325:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | |   | | |-CapturedRecordAttr [[ADDR_326:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_327:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_328:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | |   | | |-FieldDecl [[ADDR_329:0x[a-z0-9]*]] <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | |   | | `-FieldDecl [[ADDR_330:0x[a-z0-9]*]] <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |   | `-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |   |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   |   | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       | | | |   |   | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | |   |   | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | |   |   |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       | | | |   |   |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |   |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   |   |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   |   `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |     `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |-DeclRefExpr [[ADDR_331:0x[a-z0-9]*]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_332:0x[a-z0-9]*]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_333:0x[a-z0-9]*]] <line:27:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_334:0x[a-z0-9]*]] <line:28:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_335:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl [[ADDR_336:0x[a-z0-9]*]] <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_337:0x[a-z0-9]*]] <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | |-CapturedDecl [[ADDR_287]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |-CapturedStmt [[ADDR_288]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | | | |-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | | | | | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       | | | | | | | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | | | | |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       | | | | | | |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | | | | |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | | | | |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_316]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_317]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_318]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_319]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | |-DeclRefExpr [[ADDR_320]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | | | `-DeclRefExpr [[ADDR_321]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_322]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_323]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |-ImplicitParamDecl [[ADDR_324]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | | |-RecordDecl [[ADDR_325]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | |-CapturedRecordAttr [[ADDR_326]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_327]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_328]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       | | | | | |-FieldDecl [[ADDR_329]] <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       | | | | | `-FieldDecl [[ADDR_330]] <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       | | | | `-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |   |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       | | | |   | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       | | | |   | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |   | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | |   | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       | | | |   | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       | | | |   |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       | | | |   |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |   |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | |   |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | |   |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | |   |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | |   |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       | | | |   |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | |   |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       | | | |   |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | |   | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | |   `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | |     `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl [[ADDR_338:0x[a-z0-9]*]] <line:29:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | | |-OMPCapturedExprDecl [[ADDR_339:0x[a-z0-9]*]] <line:30:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       | | | | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | | | `-OMPCapturedExprDecl [[ADDR_340:0x[a-z0-9]*]] <line:29:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT: |       | | |   `-BinaryOperator [[ADDR_341:0x[a-z0-9]*]] <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT: |       | | |     |-BinaryOperator [[ADDR_342:0x[a-z0-9]*]] <col:3, line:30:28> 'long' '*'
// CHECK-NEXT: |       | | |     | |-ImplicitCastExpr [[ADDR_343:0x[a-z0-9]*]] <line:29:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT: |       | | |     | | `-BinaryOperator [[ADDR_344:0x[a-z0-9]*]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       | | |     | |   |-ParenExpr [[ADDR_345:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | |   | `-BinaryOperator [[ADDR_346:0x[a-z0-9]*]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |   |-ImplicitCastExpr [[ADDR_347:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   |   | `-DeclRefExpr [[ADDR_348:0x[a-z0-9]*]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_338]] '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     | |   |   `-ParenExpr [[ADDR_349:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT: |       | | |     | |   |     `-BinaryOperator [[ADDR_350:0x[a-z0-9]*]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       | | |     | |   |       |-BinaryOperator [[ADDR_351:0x[a-z0-9]*]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       | | |     | |   |       | |-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |   |       | `-IntegerLiteral [[ADDR_352:0x[a-z0-9]*]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | |   |       `-IntegerLiteral [[ADDR_353:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral [[ADDR_352]] <col:26> 'int' 1
// CHECK-NEXT: |       | | |     | `-ImplicitCastExpr [[ADDR_354:0x[a-z0-9]*]] <line:30:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT: |       | | |     |   `-BinaryOperator [[ADDR_355:0x[a-z0-9]*]] <col:5, col:28> 'int' '/'
// CHECK-NEXT: |       | | |     |     |-ParenExpr [[ADDR_356:0x[a-z0-9]*]] <col:5> 'int'
// CHECK-NEXT: |       | | |     |     | `-BinaryOperator [[ADDR_357:0x[a-z0-9]*]] <col:25, col:5> 'int' '-'
// CHECK-NEXT: |       | | |     |     |   |-ImplicitCastExpr [[ADDR_358:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |     |   | `-DeclRefExpr [[ADDR_359:0x[a-z0-9]*]] <col:25> 'int' {{.*}}OMPCapturedExpr [[ADDR_339]] '.capture_expr.' 'int'
// CHECK-NEXT: |       | | |     |     |   `-ParenExpr [[ADDR_360:0x[a-z0-9]*]] <col:5> 'int'
// CHECK-NEXT: |       | | |     |     |     `-BinaryOperator [[ADDR_361:0x[a-z0-9]*]] <col:18, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       | | |     |     |       |-BinaryOperator [[ADDR_362:0x[a-z0-9]*]] <col:18, col:28> 'int' '-'
// CHECK-NEXT: |       | | |     |     |       | |-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |     |     |       | `-IntegerLiteral [[ADDR_363:0x[a-z0-9]*]] <col:28> 'int' 1
// CHECK-NEXT: |       | | |     |     |       `-IntegerLiteral [[ADDR_364:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |     |     `-IntegerLiteral [[ADDR_363]] <col:28> 'int' 1
// CHECK-NEXT: |       | | |     `-ImplicitCastExpr [[ADDR_365:0x[a-z0-9]*]] <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT: |       | | |       `-IntegerLiteral [[ADDR_366:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       | | |-DeclRefExpr [[ADDR_367:0x[a-z0-9]*]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_368:0x[a-z0-9]*]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_369:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_370:0x[a-z0-9]*]] <line:27:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_371:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_372:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_373:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_374:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_375:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_376:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_377:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl [[ADDR_378:0x[a-z0-9]*]] <line:29:23> col:23 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr [[ADDR_379:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_380:0x[a-z0-9]*]] <line:30:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_381:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_281]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_282]] <line:28:1, col:59>
// CHECK-NEXT: |       |   | |-OMPCollapseClause [[ADDR_283]] <col:48, col:58>
// CHECK-NEXT: |       |   | | `-ConstantExpr [[ADDR_284]] <col:57> 'int'
// CHECK-NEXT: |       |   | |   |-value: Int 2
// CHECK-NEXT: |       |   | |   `-IntegerLiteral [[ADDR_285]] <col:57> 'int' 2
// CHECK-NEXT: |       |   | `-CapturedStmt [[ADDR_286]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   |-CapturedDecl [[ADDR_287]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | |-CapturedStmt [[ADDR_288]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   | | |-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   | | | |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   | | | | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       |   |   | | | | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | | | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | | | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       |   |   | | | |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       |   |   | | | |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | | | |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   | | | |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   |   | | | |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | | |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | | |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | | | |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   | | | |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | | | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | | | `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   | | |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_316]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_317]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_318]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_319]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   |   | | |-DeclRefExpr [[ADDR_320]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_321]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_322]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_323]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   | |-ImplicitParamDecl [[ADDR_324]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   | |-RecordDecl [[ADDR_325]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   |   | | |-CapturedRecordAttr [[ADDR_326]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_327]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_328]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   |   | | |-FieldDecl [[ADDR_329]] <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   |   | | `-FieldDecl [[ADDR_330]] <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |   | `-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |   |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   |   | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       |   |   |   | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   |   | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   |   |   | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       |   |   |   |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       |   |   |   |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   |   |   |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |   |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   |   |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   |   `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |     `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |-DeclRefExpr [[ADDR_331]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_332]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_333]] <line:27:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:27:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_334]] <line:28:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_335]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl [[ADDR_336]] <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_337]] <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   |-CapturedDecl [[ADDR_287]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |-CapturedStmt [[ADDR_288]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   | | |-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   | | | | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       |   | | | | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       |   | | | |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       |   | | | |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   | | | |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   | | | |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   | | | |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   | | | |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_316]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_317]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_318]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_319]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | |-DeclRefExpr [[ADDR_320]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   | | `-DeclRefExpr [[ADDR_321]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_322]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_323]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |-ImplicitParamDecl [[ADDR_324]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   | |-RecordDecl [[ADDR_325]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | |-CapturedRecordAttr [[ADDR_326]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_327]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_328]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT: |       |   | | |-FieldDecl [[ADDR_329]] <line:29:23> col:23 implicit 'int &'
// CHECK-NEXT: |       |   | | `-FieldDecl [[ADDR_330]] <line:30:25> col:25 implicit 'int &'
// CHECK-NEXT: |       |   | `-CapturedDecl [[ADDR_289]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |   |-ForStmt [[ADDR_290]] <line:29:3, line:31:7>
// CHECK-NEXT: |       |   |   | |-DeclStmt [[ADDR_291]] <line:29:8, col:17>
// CHECK-NEXT: |       |   |   | | `-VarDecl [[ADDR_292]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | |   `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   |   | |-BinaryOperator [[ADDR_294]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   |   | | |-ImplicitCastExpr [[ADDR_295]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | | | `-DeclRefExpr [[ADDR_296]] <col:19> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   |   | | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   | |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   |   | |-UnaryOperator [[ADDR_299]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   |   | | `-DeclRefExpr [[ADDR_300]] <col:26> 'int' {{.*}}Var [[ADDR_292]] 'i' 'int'
// CHECK-NEXT: |       |   |   | `-ForStmt [[ADDR_301]] <line:30:5, line:31:7>
// CHECK-NEXT: |       |   |   |   |-DeclStmt [[ADDR_302]] <line:30:10, col:19>
// CHECK-NEXT: |       |   |   |   | `-VarDecl [[ADDR_303]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |   |   |   `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   |   |   |-BinaryOperator [[ADDR_305]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   |   |   | |-ImplicitCastExpr [[ADDR_306]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   | | `-DeclRefExpr [[ADDR_307]] <col:21> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   |   |   |-UnaryOperator [[ADDR_310]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   |   |   | `-DeclRefExpr [[ADDR_311]] <col:28> 'int' {{.*}}Var [[ADDR_303]] 'i' 'int'
// CHECK-NEXT: |       |   |   |   `-NullStmt [[ADDR_312]] <line:31:7>
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_313]] <line:28:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_314]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   |   |-ImplicitParamDecl [[ADDR_315]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:28:1) *const restrict'
// CHECK-NEXT: |       |   |   |-VarDecl [[ADDR_292]] <line:29:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   |   | `-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |   |   `-VarDecl [[ADDR_303]] <line:30:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   |     `-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl [[ADDR_338]] <line:29:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr [[ADDR_297]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_298]] <col:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       |   |-OMPCapturedExprDecl [[ADDR_339]] <line:30:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr [[ADDR_308]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_309]] <col:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: |       |   `-OMPCapturedExprDecl [[ADDR_340]] <line:29:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT: |       |     `-BinaryOperator [[ADDR_341]] <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT: |       |       |-BinaryOperator [[ADDR_342]] <col:3, line:30:28> 'long' '*'
// CHECK-NEXT: |       |       | |-ImplicitCastExpr [[ADDR_343]] <line:29:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT: |       |       | | `-BinaryOperator [[ADDR_344]] <col:3, col:26> 'int' '/'
// CHECK-NEXT: |       |       | |   |-ParenExpr [[ADDR_345]] <col:3> 'int'
// CHECK-NEXT: |       |       | |   | `-BinaryOperator [[ADDR_346]] <col:23, col:3> 'int' '-'
// CHECK-NEXT: |       |       | |   |   |-ImplicitCastExpr [[ADDR_347]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   |   | `-DeclRefExpr [[ADDR_348]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_338]] '.capture_expr.' 'int'
// CHECK-NEXT: |       |       | |   |   `-ParenExpr [[ADDR_349]] <col:3> 'int'
// CHECK-NEXT: |       |       | |   |     `-BinaryOperator [[ADDR_350]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       |       | |   |       |-BinaryOperator [[ADDR_351]] <col:16, col:26> 'int' '-'
// CHECK-NEXT: |       |       | |   |       | |-IntegerLiteral [[ADDR_293]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |   |       | `-IntegerLiteral [[ADDR_352]] <col:26> 'int' 1
// CHECK-NEXT: |       |       | |   |       `-IntegerLiteral [[ADDR_353]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       | |   `-IntegerLiteral [[ADDR_352]] <col:26> 'int' 1
// CHECK-NEXT: |       |       | `-ImplicitCastExpr [[ADDR_354]] <line:30:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT: |       |       |   `-BinaryOperator [[ADDR_355]] <col:5, col:28> 'int' '/'
// CHECK-NEXT: |       |       |     |-ParenExpr [[ADDR_356]] <col:5> 'int'
// CHECK-NEXT: |       |       |     | `-BinaryOperator [[ADDR_357]] <col:25, col:5> 'int' '-'
// CHECK-NEXT: |       |       |     |   |-ImplicitCastExpr [[ADDR_358]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |     |   | `-DeclRefExpr [[ADDR_359]] <col:25> 'int' {{.*}}OMPCapturedExpr [[ADDR_339]] '.capture_expr.' 'int'
// CHECK-NEXT: |       |       |     |   `-ParenExpr [[ADDR_360]] <col:5> 'int'
// CHECK-NEXT: |       |       |     |     `-BinaryOperator [[ADDR_361]] <col:18, <invalid sloc>> 'int' '+'
// CHECK-NEXT: |       |       |     |       |-BinaryOperator [[ADDR_362]] <col:18, col:28> 'int' '-'
// CHECK-NEXT: |       |       |     |       | |-IntegerLiteral [[ADDR_304]] <col:18> 'int' 0
// CHECK-NEXT: |       |       |     |       | `-IntegerLiteral [[ADDR_363]] <col:28> 'int' 1
// CHECK-NEXT: |       |       |     |       `-IntegerLiteral [[ADDR_364]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |       |     `-IntegerLiteral [[ADDR_363]] <col:28> 'int' 1
// CHECK-NEXT: |       |       `-ImplicitCastExpr [[ADDR_365]] <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT: |       |         `-IntegerLiteral [[ADDR_366]] <<invalid sloc>> 'int' 1
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_382:0x[a-z0-9]*]] <line:29:23> 'int' {{.*}}ParmVar [[ADDR_271]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_383:0x[a-z0-9]*]] <line:30:25> 'int' {{.*}}ParmVar [[ADDR_272]] 'y' 'int'
// CHECK-NEXT: `-FunctionDecl [[ADDR_384:0x[a-z0-9]*]] <line:34:1, line:41:1> line:34:6 test_five 'void (int, int, int)'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_385:0x[a-z0-9]*]] <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_386:0x[a-z0-9]*]] <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_387:0x[a-z0-9]*]] <col:30, col:34> col:34 used z 'int'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_388:0x[a-z0-9]*]] <col:37, line:41:1>
// CHECK-NEXT:     `-OMPTargetDirective [[ADDR_389:0x[a-z0-9]*]] <line:35:1, col:19>
// CHECK-NEXT:       |-OMPFirstprivateClause [[ADDR_390:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT:       | |-DeclRefExpr [[ADDR_391:0x[a-z0-9]*]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:       | |-DeclRefExpr [[ADDR_392:0x[a-z0-9]*]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:       | `-DeclRefExpr [[ADDR_393:0x[a-z0-9]*]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:       `-CapturedStmt [[ADDR_394:0x[a-z0-9]*]] <line:36:1, col:59>
// CHECK-NEXT:         |-CapturedDecl [[ADDR_395:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | |-CapturedStmt [[ADDR_396:0x[a-z0-9]*]] <col:1, col:59>
// CHECK-NEXT:         | | |-CapturedDecl [[ADDR_397:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_398:0x[a-z0-9]*]] <col:1, col:59>
// CHECK-NEXT:         | | | | |-OMPCollapseClause [[ADDR_399:0x[a-z0-9]*]] <col:48, col:58>
// CHECK-NEXT:         | | | | | `-ConstantExpr [[ADDR_400:0x[a-z0-9]*]] <col:57> 'int'
// CHECK-NEXT:         | | | | |   |-value: Int 2
// CHECK-NEXT:         | | | | |   `-IntegerLiteral [[ADDR_401:0x[a-z0-9]*]] <col:57> 'int' 2
// CHECK-NEXT:         | | | | `-CapturedStmt [[ADDR_402:0x[a-z0-9]*]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   |-CapturedDecl [[ADDR_403:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   | |-CapturedStmt [[ADDR_404:0x[a-z0-9]*]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   | | |-CapturedDecl [[ADDR_405:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   | | | |-ForStmt [[ADDR_406:0x[a-z0-9]*]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   | | | | |-DeclStmt [[ADDR_407:0x[a-z0-9]*]] <line:37:8, col:17>
// CHECK-NEXT:         | | | |   | | | | | `-VarDecl [[ADDR_408:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | | |   `-IntegerLiteral [[ADDR_409:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT:         | | | |   | | | | |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | | | | |-BinaryOperator [[ADDR_410:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |   | | | | | |-ImplicitCastExpr [[ADDR_411:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | | | | `-DeclRefExpr [[ADDR_412:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | | | | `-ImplicitCastExpr [[ADDR_413:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | | |   `-DeclRefExpr [[ADDR_414:0x[a-z0-9]*]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | |   | | | | |-UnaryOperator [[ADDR_415:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | | | | `-DeclRefExpr [[ADDR_416:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | | | `-ForStmt [[ADDR_417:0x[a-z0-9]*]] <line:38:5, line:40:9>
// CHECK-NEXT:         | | | |   | | | |   |-DeclStmt [[ADDR_418:0x[a-z0-9]*]] <line:38:10, col:19>
// CHECK-NEXT:         | | | |   | | | |   | `-VarDecl [[ADDR_419:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | |   |   `-IntegerLiteral [[ADDR_420:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT:         | | | |   | | | |   |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | | | |   |-BinaryOperator [[ADDR_421:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | |   | | | |   | |-ImplicitCastExpr [[ADDR_422:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |   | | `-DeclRefExpr [[ADDR_423:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |   | `-ImplicitCastExpr [[ADDR_424:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |   |   `-DeclRefExpr [[ADDR_425:0x[a-z0-9]*]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | |   | | | |   |-UnaryOperator [[ADDR_426:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | | |   | `-DeclRefExpr [[ADDR_427:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |   `-ForStmt [[ADDR_428:0x[a-z0-9]*]] <line:39:7, line:40:9>
// CHECK-NEXT:         | | | |   | | | |     |-DeclStmt [[ADDR_429:0x[a-z0-9]*]] <line:39:12, col:21>
// CHECK-NEXT:         | | | |   | | | |     | `-VarDecl [[ADDR_430:0x[a-z0-9]*]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | |     |   `-IntegerLiteral [[ADDR_431:0x[a-z0-9]*]] <col:20> 'int' 0
// CHECK-NEXT:         | | | |   | | | |     |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | | | |     |-BinaryOperator [[ADDR_432:0x[a-z0-9]*]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | |   | | | |     | |-ImplicitCastExpr [[ADDR_433:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |     | | `-DeclRefExpr [[ADDR_434:0x[a-z0-9]*]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |     | `-ImplicitCastExpr [[ADDR_435:0x[a-z0-9]*]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | |     |   `-DeclRefExpr [[ADDR_436:0x[a-z0-9]*]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | |   | | | |     |-UnaryOperator [[ADDR_437:0x[a-z0-9]*]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | | |     | `-DeclRefExpr [[ADDR_438:0x[a-z0-9]*]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | | |     `-NullStmt [[ADDR_439:0x[a-z0-9]*]] <line:40:9>
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl [[ADDR_440:0x[a-z0-9]*]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl [[ADDR_441:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | | | |-ImplicitParamDecl [[ADDR_442:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   | | | |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | |   | | | |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | | | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | |   | | | `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   | | |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | |   | | |-DeclRefExpr [[ADDR_443:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_444:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         | | | |   | | |-DeclRefExpr [[ADDR_445:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_446:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         | | | |   | | |-DeclRefExpr [[ADDR_447:0x[a-z0-9]*]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | |   | | |-DeclRefExpr [[ADDR_448:0x[a-z0-9]*]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | |   | | `-DeclRefExpr [[ADDR_449:0x[a-z0-9]*]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | |   | |-ImplicitParamDecl [[ADDR_450:0x[a-z0-9]*]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | |-ImplicitParamDecl [[ADDR_451:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   | |-ImplicitParamDecl [[ADDR_452:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   | |-RecordDecl [[ADDR_453:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | |   | | |-CapturedRecordAttr [[ADDR_454:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | |   | | |-FieldDecl [[ADDR_455:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         | | | |   | | |-FieldDecl [[ADDR_456:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         | | | |   | | |-FieldDecl [[ADDR_457:0x[a-z0-9]*]] <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         | | | |   | | |-FieldDecl [[ADDR_458:0x[a-z0-9]*]] <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         | | | |   | | `-FieldDecl [[ADDR_459:0x[a-z0-9]*]] <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         | | | |   | `-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   |   |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   |   | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         | | | |   |   | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | |   |   | |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |   |   | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | |   |   | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         | | | |   |   |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         | | | |   |   |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | |   |   |   |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | |   |   |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | |   |   |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         | | | |   |   |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         | | | |   |   |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | |   |   |     |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | |   |   |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | |   |   |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |   |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   |   |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | |   |   |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | |   |   `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   |     `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | |   |-DeclRefExpr [[ADDR_460:0x[a-z0-9]*]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | |   |-DeclRefExpr [[ADDR_461:0x[a-z0-9]*]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | |   `-DeclRefExpr [[ADDR_462:0x[a-z0-9]*]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | |-ImplicitParamDecl [[ADDR_463:0x[a-z0-9]*]] <line:35:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK-NEXT:         | | | |-RecordDecl [[ADDR_464:0x[a-z0-9]*]] <line:36:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | | |-CapturedRecordAttr [[ADDR_465:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | | |-FieldDecl [[ADDR_466:0x[a-z0-9]*]] <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         | | | | |-FieldDecl [[ADDR_467:0x[a-z0-9]*]] <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         | | | | `-FieldDecl [[ADDR_468:0x[a-z0-9]*]] <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         | | | |-CapturedDecl [[ADDR_403]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | |-CapturedStmt [[ADDR_404]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | | | |-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | | | |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | | | | | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         | | | | | | | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | | | | | |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | | | | | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | | | | | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         | | | | | | |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         | | | | | | |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | | | |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | | | | |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | | | | |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | | | | |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | | | | |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         | | | | | | |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         | | | | | | |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | | | |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | | | | |     |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | | | | |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | | | | |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | | | | |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | | | | |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | | |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | | | | |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | | | | |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | | | | `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | | |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | | | |-DeclRefExpr [[ADDR_443]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_444]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         | | | | | |-DeclRefExpr [[ADDR_445]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_446]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         | | | | | |-DeclRefExpr [[ADDR_447]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | | | |-DeclRefExpr [[ADDR_448]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | | | `-DeclRefExpr [[ADDR_449]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | | |-ImplicitParamDecl [[ADDR_450]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | |-ImplicitParamDecl [[ADDR_451]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | |-ImplicitParamDecl [[ADDR_452]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | | |-RecordDecl [[ADDR_453]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | | | |-CapturedRecordAttr [[ADDR_454]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | | | |-FieldDecl [[ADDR_455]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         | | | | | |-FieldDecl [[ADDR_456]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         | | | | | |-FieldDecl [[ADDR_457]] <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         | | | | | |-FieldDecl [[ADDR_458]] <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         | | | | | `-FieldDecl [[ADDR_459]] <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         | | | | `-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |   |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         | | | |   | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         | | | |   | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | |   | |-<<<NULL>>>
// CHECK-NEXT:         | | | |   | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |   | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | |   | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | |   | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | |   | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         | | | |   | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         | | | |   |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         | | | |   |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | |   |   |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | |   |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | |   |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         | | | |   |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         | | | |   |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         | | | |   |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |   |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | |   |     |-<<<NULL>>>
// CHECK-NEXT:         | | | |   |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | |   |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | |   |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | | | |   |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | |   |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         | | | |   |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | |   |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         | | | |   |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | |   | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | | |   |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | |   | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | | |   `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | |     `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         | | | |-OMPCapturedExprDecl [[ADDR_469:0x[a-z0-9]*]] <line:37:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | | |-OMPCapturedExprDecl [[ADDR_470:0x[a-z0-9]*]] <line:38:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT:         | | | | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | | `-OMPCapturedExprDecl [[ADDR_471:0x[a-z0-9]*]] <line:37:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT:         | | |   `-BinaryOperator [[ADDR_472:0x[a-z0-9]*]] <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT:         | | |     |-BinaryOperator [[ADDR_473:0x[a-z0-9]*]] <col:3, line:38:28> 'long' '*'
// CHECK-NEXT:         | | |     | |-ImplicitCastExpr [[ADDR_474:0x[a-z0-9]*]] <line:37:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT:         | | |     | | `-BinaryOperator [[ADDR_475:0x[a-z0-9]*]] <col:3, col:26> 'int' '/'
// CHECK-NEXT:         | | |     | |   |-ParenExpr [[ADDR_476:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT:         | | |     | |   | `-BinaryOperator [[ADDR_477:0x[a-z0-9]*]] <col:23, col:3> 'int' '-'
// CHECK-NEXT:         | | |     | |   |   |-ImplicitCastExpr [[ADDR_478:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     | |   |   | `-DeclRefExpr [[ADDR_479:0x[a-z0-9]*]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_469]] '.capture_expr.' 'int'
// CHECK-NEXT:         | | |     | |   |   `-ParenExpr [[ADDR_480:0x[a-z0-9]*]] <col:3> 'int'
// CHECK-NEXT:         | | |     | |   |     `-BinaryOperator [[ADDR_481:0x[a-z0-9]*]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT:         | | |     | |   |       |-BinaryOperator [[ADDR_482:0x[a-z0-9]*]] <col:16, col:26> 'int' '-'
// CHECK-NEXT:         | | |     | |   |       | |-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         | | |     | |   |       | `-IntegerLiteral [[ADDR_483:0x[a-z0-9]*]] <col:26> 'int' 1
// CHECK-NEXT:         | | |     | |   |       `-IntegerLiteral [[ADDR_484:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT:         | | |     | |   `-IntegerLiteral [[ADDR_483]] <col:26> 'int' 1
// CHECK-NEXT:         | | |     | `-ImplicitCastExpr [[ADDR_485:0x[a-z0-9]*]] <line:38:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT:         | | |     |   `-BinaryOperator [[ADDR_486:0x[a-z0-9]*]] <col:5, col:28> 'int' '/'
// CHECK-NEXT:         | | |     |     |-ParenExpr [[ADDR_487:0x[a-z0-9]*]] <col:5> 'int'
// CHECK-NEXT:         | | |     |     | `-BinaryOperator [[ADDR_488:0x[a-z0-9]*]] <col:25, col:5> 'int' '-'
// CHECK-NEXT:         | | |     |     |   |-ImplicitCastExpr [[ADDR_489:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     |     |   | `-DeclRefExpr [[ADDR_490:0x[a-z0-9]*]] <col:25> 'int' {{.*}}OMPCapturedExpr [[ADDR_470]] '.capture_expr.' 'int'
// CHECK-NEXT:         | | |     |     |   `-ParenExpr [[ADDR_491:0x[a-z0-9]*]] <col:5> 'int'
// CHECK-NEXT:         | | |     |     |     `-BinaryOperator [[ADDR_492:0x[a-z0-9]*]] <col:18, <invalid sloc>> 'int' '+'
// CHECK-NEXT:         | | |     |     |       |-BinaryOperator [[ADDR_493:0x[a-z0-9]*]] <col:18, col:28> 'int' '-'
// CHECK-NEXT:         | | |     |     |       | |-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         | | |     |     |       | `-IntegerLiteral [[ADDR_494:0x[a-z0-9]*]] <col:28> 'int' 1
// CHECK-NEXT:         | | |     |     |       `-IntegerLiteral [[ADDR_495:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT:         | | |     |     `-IntegerLiteral [[ADDR_494]] <col:28> 'int' 1
// CHECK-NEXT:         | | |     `-ImplicitCastExpr [[ADDR_496:0x[a-z0-9]*]] <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT:         | | |       `-IntegerLiteral [[ADDR_497:0x[a-z0-9]*]] <<invalid sloc>> 'int' 1
// CHECK-NEXT:         | | |-DeclRefExpr [[ADDR_498:0x[a-z0-9]*]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         | | |-DeclRefExpr [[ADDR_499:0x[a-z0-9]*]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         | | `-DeclRefExpr [[ADDR_500:0x[a-z0-9]*]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         | |-AlwaysInlineAttr [[ADDR_501:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_502:0x[a-z0-9]*]] <line:35:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_503:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_504:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_505:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_506:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_507:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK-NEXT:         | |-RecordDecl [[ADDR_508:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | |-CapturedRecordAttr [[ADDR_509:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | |-FieldDecl [[ADDR_510:0x[a-z0-9]*]] <line:37:23> col:23 implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr [[ADDR_511:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |-FieldDecl [[ADDR_512:0x[a-z0-9]*]] <line:38:25> col:25 implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr [[ADDR_513:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | `-FieldDecl [[ADDR_514:0x[a-z0-9]*]] <line:39:27> col:27 implicit 'int'
// CHECK-NEXT:         | |   `-OMPCaptureKindAttr [[ADDR_515:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | `-CapturedDecl [[ADDR_397]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |-OMPTeamsDistributeParallelForSimdDirective [[ADDR_398]] <line:36:1, col:59>
// CHECK-NEXT:         |   | |-OMPCollapseClause [[ADDR_399]] <col:48, col:58>
// CHECK-NEXT:         |   | | `-ConstantExpr [[ADDR_400]] <col:57> 'int'
// CHECK-NEXT:         |   | |   |-value: Int 2
// CHECK-NEXT:         |   | |   `-IntegerLiteral [[ADDR_401]] <col:57> 'int' 2
// CHECK-NEXT:         |   | `-CapturedStmt [[ADDR_402]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   |-CapturedDecl [[ADDR_403]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   | |-CapturedStmt [[ADDR_404]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   | | |-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   | | | |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   | | | | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         |   |   | | | | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   |   | | | | |-<<<NULL>>>
// CHECK-NEXT:         |   |   | | | | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   |   | | | | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   |   | | | | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   |   | | | | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | | | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   |   | | | | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         |   |   | | | |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         |   |   | | | |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   |   | | | |   |-<<<NULL>>>
// CHECK-NEXT:         |   |   | | | |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   |   | | | |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   |   | | | |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   |   | | | |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | | |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   |   | | | |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         |   |   | | | |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         |   |   | | | |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   |   | | | |     |-<<<NULL>>>
// CHECK-NEXT:         |   |   | | | |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   |   | | | |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   |   | | | |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   |   | | | |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | | |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   |   | | | |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | | | |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   | | | |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   |   | | | |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   | | | | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   |   | | | `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   | | |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   |   | | |-DeclRefExpr [[ADDR_443]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_444]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         |   |   | | |-DeclRefExpr [[ADDR_445]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_446]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         |   |   | | |-DeclRefExpr [[ADDR_447]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   |   | | |-DeclRefExpr [[ADDR_448]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   |   | | `-DeclRefExpr [[ADDR_449]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   |   | |-ImplicitParamDecl [[ADDR_450]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | |-ImplicitParamDecl [[ADDR_451]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   | |-ImplicitParamDecl [[ADDR_452]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   | |-RecordDecl [[ADDR_453]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |   |   | | |-CapturedRecordAttr [[ADDR_454]] <<invalid sloc>> Implicit
// CHECK-NEXT:         |   |   | | |-FieldDecl [[ADDR_455]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         |   |   | | |-FieldDecl [[ADDR_456]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         |   |   | | |-FieldDecl [[ADDR_457]] <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         |   |   | | |-FieldDecl [[ADDR_458]] <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         |   |   | | `-FieldDecl [[ADDR_459]] <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         |   |   | `-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   |   |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   |   | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         |   |   |   | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   |   | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   |   |   | |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   |   |   | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   |   |   | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   |   |   | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   |   |   | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         |   |   |   |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         |   |   |   |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   |   |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   |   |   |   |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   |   |   |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   |   |   |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   |   |   |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   |   |   |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         |   |   |   |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         |   |   |   |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   |   |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   |   |   |     |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   |   |   |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   |   |   |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   |   |   |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   |   |   |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |   |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   |   |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   |   | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   |   |   |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   |   | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   |   |   `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   |     `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   |   |-DeclRefExpr [[ADDR_460]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   |   |-DeclRefExpr [[ADDR_461]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   |   `-DeclRefExpr [[ADDR_462]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   |-ImplicitParamDecl [[ADDR_463]] <line:35:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:35:1) *const restrict'
// CHECK-NEXT:         |   |-RecordDecl [[ADDR_464]] <line:36:1> col:1 implicit struct definition
// CHECK-NEXT:         |   | |-CapturedRecordAttr [[ADDR_465]] <<invalid sloc>> Implicit
// CHECK-NEXT:         |   | |-FieldDecl [[ADDR_466]] <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         |   | |-FieldDecl [[ADDR_467]] <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         |   | `-FieldDecl [[ADDR_468]] <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         |   |-CapturedDecl [[ADDR_403]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | |-CapturedStmt [[ADDR_404]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   | | |-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | | | |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   | | | | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         |   | | | | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | | | | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   | | | | |-<<<NULL>>>
// CHECK-NEXT:         |   | | | | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   | | | | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   | | | | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   | | | | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   | | | | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   | | | | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         |   | | | |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         |   | | | |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | | | |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   | | | |   |-<<<NULL>>>
// CHECK-NEXT:         |   | | | |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   | | | |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   | | | |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   | | | |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   | | | |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   | | | |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         |   | | | |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         |   | | | |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | | | |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   | | | |     |-<<<NULL>>>
// CHECK-NEXT:         |   | | | |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   | | | |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   | | | |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   | | | |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   | | | |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   | | | |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | | |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   | | | |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | | | | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   | | | |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | | | | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   | | | `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | | |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   | | |-DeclRefExpr [[ADDR_443]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_444]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         |   | | |-DeclRefExpr [[ADDR_445]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_446]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         |   | | |-DeclRefExpr [[ADDR_447]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   | | |-DeclRefExpr [[ADDR_448]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   | | `-DeclRefExpr [[ADDR_449]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   | |-ImplicitParamDecl [[ADDR_450]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | |-ImplicitParamDecl [[ADDR_451]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | |-ImplicitParamDecl [[ADDR_452]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   | |-RecordDecl [[ADDR_453]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |   | | |-CapturedRecordAttr [[ADDR_454]] <<invalid sloc>> Implicit
// CHECK-NEXT:         |   | | |-FieldDecl [[ADDR_455]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         |   | | |-FieldDecl [[ADDR_456]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long &'
// CHECK-NEXT:         |   | | |-FieldDecl [[ADDR_457]] <line:37:23> col:23 implicit 'int &'
// CHECK-NEXT:         |   | | |-FieldDecl [[ADDR_458]] <line:38:25> col:25 implicit 'int &'
// CHECK-NEXT:         |   | | `-FieldDecl [[ADDR_459]] <line:39:27> col:27 implicit 'int &'
// CHECK-NEXT:         |   | `-CapturedDecl [[ADDR_405]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |   |-ForStmt [[ADDR_406]] <line:37:3, line:40:9>
// CHECK-NEXT:         |   |   | |-DeclStmt [[ADDR_407]] <line:37:8, col:17>
// CHECK-NEXT:         |   |   | | `-VarDecl [[ADDR_408]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | |   `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   |   | |-<<<NULL>>>
// CHECK-NEXT:         |   |   | |-BinaryOperator [[ADDR_410]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   |   | | |-ImplicitCastExpr [[ADDR_411]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | | | `-DeclRefExpr [[ADDR_412]] <col:19> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   |   | | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   | |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   |   | |-UnaryOperator [[ADDR_415]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   |   | | `-DeclRefExpr [[ADDR_416]] <col:26> 'int' {{.*}}Var [[ADDR_408]] 'i' 'int'
// CHECK-NEXT:         |   |   | `-ForStmt [[ADDR_417]] <line:38:5, line:40:9>
// CHECK-NEXT:         |   |   |   |-DeclStmt [[ADDR_418]] <line:38:10, col:19>
// CHECK-NEXT:         |   |   |   | `-VarDecl [[ADDR_419]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   |   |   `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   |   |   |-<<<NULL>>>
// CHECK-NEXT:         |   |   |   |-BinaryOperator [[ADDR_421]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   |   |   | |-ImplicitCastExpr [[ADDR_422]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   | | `-DeclRefExpr [[ADDR_423]] <col:21> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   |   |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   |   |   |-UnaryOperator [[ADDR_426]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   |   |   | `-DeclRefExpr [[ADDR_427]] <col:28> 'int' {{.*}}Var [[ADDR_419]] 'i' 'int'
// CHECK-NEXT:         |   |   |   `-ForStmt [[ADDR_428]] <line:39:7, line:40:9>
// CHECK-NEXT:         |   |   |     |-DeclStmt [[ADDR_429]] <line:39:12, col:21>
// CHECK-NEXT:         |   |   |     | `-VarDecl [[ADDR_430]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |   |     |   `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   |   |     |-<<<NULL>>>
// CHECK-NEXT:         |   |   |     |-BinaryOperator [[ADDR_432]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   |   |     | |-ImplicitCastExpr [[ADDR_433]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |     | | `-DeclRefExpr [[ADDR_434]] <col:23> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   |   |     | `-ImplicitCastExpr [[ADDR_435]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   |     |   `-DeclRefExpr [[ADDR_436]] <col:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
// CHECK-NEXT:         |   |   |     |-UnaryOperator [[ADDR_437]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   |   |     | `-DeclRefExpr [[ADDR_438]] <col:30> 'int' {{.*}}Var [[ADDR_430]] 'i' 'int'
// CHECK-NEXT:         |   |   |     `-NullStmt [[ADDR_439]] <line:40:9>
// CHECK-NEXT:         |   |   |-ImplicitParamDecl [[ADDR_440]] <line:36:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |-ImplicitParamDecl [[ADDR_441]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   |   |-ImplicitParamDecl [[ADDR_442]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-teams-distribute-parallel-for-simd.c:36:1) *const restrict'
// CHECK-NEXT:         |   |   |-VarDecl [[ADDR_408]] <line:37:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   |   | `-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |   |   |-VarDecl [[ADDR_419]] <line:38:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   |   | `-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |   |   `-VarDecl [[ADDR_430]] <line:39:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   |     `-IntegerLiteral [[ADDR_431]] <col:20> 'int' 0
// CHECK-NEXT:         |   |-OMPCapturedExprDecl [[ADDR_469]] <line:37:23> col:23 implicit used .capture_expr. 'int'
// CHECK-NEXT:         |   | `-ImplicitCastExpr [[ADDR_413]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   `-DeclRefExpr [[ADDR_414]] <col:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |   |-OMPCapturedExprDecl [[ADDR_470]] <line:38:25> col:25 implicit used .capture_expr. 'int'
// CHECK-NEXT:         |   | `-ImplicitCastExpr [[ADDR_424]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   |   `-DeclRefExpr [[ADDR_425]] <col:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         |   `-OMPCapturedExprDecl [[ADDR_471]] <line:37:3, <invalid sloc>> col:3 implicit used .capture_expr. 'long'
// CHECK-NEXT:         |     `-BinaryOperator [[ADDR_472]] <col:3, <invalid sloc>> 'long' '-'
// CHECK-NEXT:         |       |-BinaryOperator [[ADDR_473]] <col:3, line:38:28> 'long' '*'
// CHECK-NEXT:         |       | |-ImplicitCastExpr [[ADDR_474]] <line:37:3, col:26> 'long' <IntegralCast>
// CHECK-NEXT:         |       | | `-BinaryOperator [[ADDR_475]] <col:3, col:26> 'int' '/'
// CHECK-NEXT:         |       | |   |-ParenExpr [[ADDR_476]] <col:3> 'int'
// CHECK-NEXT:         |       | |   | `-BinaryOperator [[ADDR_477]] <col:23, col:3> 'int' '-'
// CHECK-NEXT:         |       | |   |   |-ImplicitCastExpr [[ADDR_478]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |       | |   |   | `-DeclRefExpr [[ADDR_479]] <col:23> 'int' {{.*}}OMPCapturedExpr [[ADDR_469]] '.capture_expr.' 'int'
// CHECK-NEXT:         |       | |   |   `-ParenExpr [[ADDR_480]] <col:3> 'int'
// CHECK-NEXT:         |       | |   |     `-BinaryOperator [[ADDR_481]] <col:16, <invalid sloc>> 'int' '+'
// CHECK-NEXT:         |       | |   |       |-BinaryOperator [[ADDR_482]] <col:16, col:26> 'int' '-'
// CHECK-NEXT:         |       | |   |       | |-IntegerLiteral [[ADDR_409]] <col:16> 'int' 0
// CHECK-NEXT:         |       | |   |       | `-IntegerLiteral [[ADDR_483]] <col:26> 'int' 1
// CHECK-NEXT:         |       | |   |       `-IntegerLiteral [[ADDR_484]] <<invalid sloc>> 'int' 1
// CHECK-NEXT:         |       | |   `-IntegerLiteral [[ADDR_483]] <col:26> 'int' 1
// CHECK-NEXT:         |       | `-ImplicitCastExpr [[ADDR_485]] <line:38:5, col:28> 'long' <IntegralCast>
// CHECK-NEXT:         |       |   `-BinaryOperator [[ADDR_486]] <col:5, col:28> 'int' '/'
// CHECK-NEXT:         |       |     |-ParenExpr [[ADDR_487]] <col:5> 'int'
// CHECK-NEXT:         |       |     | `-BinaryOperator [[ADDR_488]] <col:25, col:5> 'int' '-'
// CHECK-NEXT:         |       |     |   |-ImplicitCastExpr [[ADDR_489]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |       |     |   | `-DeclRefExpr [[ADDR_490]] <col:25> 'int' {{.*}}OMPCapturedExpr [[ADDR_470]] '.capture_expr.' 'int'
// CHECK-NEXT:         |       |     |   `-ParenExpr [[ADDR_491]] <col:5> 'int'
// CHECK-NEXT:         |       |     |     `-BinaryOperator [[ADDR_492]] <col:18, <invalid sloc>> 'int' '+'
// CHECK-NEXT:         |       |     |       |-BinaryOperator [[ADDR_493]] <col:18, col:28> 'int' '-'
// CHECK-NEXT:         |       |     |       | |-IntegerLiteral [[ADDR_420]] <col:18> 'int' 0
// CHECK-NEXT:         |       |     |       | `-IntegerLiteral [[ADDR_494]] <col:28> 'int' 1
// CHECK-NEXT:         |       |     |       `-IntegerLiteral [[ADDR_495]] <<invalid sloc>> 'int' 1
// CHECK-NEXT:         |       |     `-IntegerLiteral [[ADDR_494]] <col:28> 'int' 1
// CHECK-NEXT:         |       `-ImplicitCastExpr [[ADDR_496]] <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT:         |         `-IntegerLiteral [[ADDR_497]] <<invalid sloc>> 'int' 1
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_516:0x[a-z0-9]*]] <line:37:23> 'int' {{.*}}ParmVar [[ADDR_385]] 'x' 'int'
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_517:0x[a-z0-9]*]] <line:38:25> 'int' {{.*}}ParmVar [[ADDR_386]] 'y' 'int'
// CHECK-NEXT:         `-DeclRefExpr [[ADDR_518:0x[a-z0-9]*]] <line:39:27> 'int' {{.*}}ParmVar [[ADDR_387]] 'z' 'int'
