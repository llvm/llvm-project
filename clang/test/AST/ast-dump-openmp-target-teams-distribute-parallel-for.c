// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(int x) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
#pragma omp target teams distribute parallel for collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
#pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
#pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:3:6 test_one 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_1:0x[a-z0-9]*]] <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_2:0x[a-z0-9]*]] <col:22, line:7:1>
// CHECK-NEXT: |   `-OMPTargetTeamsDistributeParallelForDirective [[ADDR_3:0x[a-z0-9]*]] <line:4:1, col:49>
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:5:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_6:0x[a-z0-9]*]] <col:3, line:6:5>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_7:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_8:0x[a-z0-9]*]] <line:5:3, line:6:5>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_9:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-CapturedStmt [[ADDR_10:0x[a-z0-9]*]] <line:5:3, line:6:5>
// CHECK-NEXT: |       | | | | |-CapturedDecl [[ADDR_11:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | |-CapturedStmt [[ADDR_12:0x[a-z0-9]*]] <line:5:3, line:6:5>
// CHECK-NEXT: |       | | | | | | |-CapturedDecl [[ADDR_13:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | | |-ForStmt [[ADDR_14:0x[a-z0-9]*]] <line:5:3, line:6:5>
// CHECK-NEXT: |       | | | | | | | | |-DeclStmt [[ADDR_15:0x[a-z0-9]*]] <line:5:8, col:17>
// CHECK-NEXT: |       | | | | | | | | | `-VarDecl [[ADDR_16:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | |   `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | | |-BinaryOperator [[ADDR_18:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | | |-ImplicitCastExpr [[ADDR_19:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | | `-DeclRefExpr [[ADDR_20:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | | `-ImplicitCastExpr [[ADDR_21:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | | |-UnaryOperator [[ADDR_22:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_23:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-NullStmt [[ADDR_24:0x[a-z0-9]*]] <line:6:5>
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_25:0x[a-z0-9]*]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_26:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_27:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | | | | | | `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_28:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_29:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_30:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | `-DeclRefExpr [[ADDR_32:0x[a-z0-9]*]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_33:0x[a-z0-9]*]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_34:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_35:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | | | | |-RecordDecl [[ADDR_36:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | | |-CapturedRecordAttr [[ADDR_37:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_38:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_39:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_40:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_41:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | `-FieldDecl [[ADDR_42:0x[a-z0-9]*]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | |   `-OMPCaptureKindAttr [[ADDR_43:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | `-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |   |-ForStmt [[ADDR_14]] <col:3, line:6:5>
// CHECK-NEXT: |       | | | | |   | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       | | | | |   | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | |   | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | | |   | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | | | |   `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |     `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_44:0x[a-z0-9]*]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_45:0x[a-z0-9]*]] <line:4:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_46:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_47:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_48:0x[a-z0-9]*]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | |   `-OMPCaptureKindAttr [[ADDR_49:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | `-CapturedDecl [[ADDR_11]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   |-CapturedStmt [[ADDR_12]] <col:3, line:6:5>
// CHECK-NEXT: |       | | |   | |-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   | | |-ForStmt [[ADDR_14]] <line:5:3, line:6:5>
// CHECK-NEXT: |       | | |   | | | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       | | |   | | | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |   | | | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | |   | | | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | |   | | `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_28]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_29]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_30]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | |   | `-DeclRefExpr [[ADDR_32]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_33]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_34]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_35]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | |   |-RecordDecl [[ADDR_36]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |   | |-CapturedRecordAttr [[ADDR_37]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_38]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_39]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_40]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_41]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | `-FieldDecl [[ADDR_42]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | |   |   `-OMPCaptureKindAttr [[ADDR_43]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   `-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |     |-ForStmt [[ADDR_14]] <col:3, line:6:5>
// CHECK-NEXT: |       | | |     | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       | | |     | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |-<<<NULL>>>
// CHECK-NEXT: |       | | |     | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |     | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | |     | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | | |     | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |     | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       | | |     | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | |     `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |       `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_50:0x[a-z0-9]*]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_51:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_52:0x[a-z0-9]*]] <line:4:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_53:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_54:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_55:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_56:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_57:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_58:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_59:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_60:0x[a-z0-9]*]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_61:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_9]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-CapturedStmt [[ADDR_10]] <col:3, line:6:5>
// CHECK-NEXT: |       |   | |-CapturedDecl [[ADDR_11]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | |-CapturedStmt [[ADDR_12]] <line:5:3, line:6:5>
// CHECK-NEXT: |       |   | | | |-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | | |-ForStmt [[ADDR_14]] <line:5:3, line:6:5>
// CHECK-NEXT: |       |   | | | | | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       |   | | | | | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |   | | | | `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_28]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_29]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_30]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | | `-DeclRefExpr [[ADDR_32]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_33]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_34]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_35]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |   | | |-RecordDecl [[ADDR_36]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | | |-CapturedRecordAttr [[ADDR_37]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_38]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_39]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_40]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_41]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | `-FieldDecl [[ADDR_42]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | |   `-OMPCaptureKindAttr [[ADDR_43]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | `-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |   |-ForStmt [[ADDR_14]] <col:3, line:6:5>
// CHECK-NEXT: |       |   | |   | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       |   | |   | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | |   | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   | |   | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |   | |   `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |     `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | `-DeclRefExpr [[ADDR_44]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_45]] <line:4:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_46]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_47]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_48]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   |   `-OMPCaptureKindAttr [[ADDR_49]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   `-CapturedDecl [[ADDR_11]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     |-CapturedStmt [[ADDR_12]] <col:3, line:6:5>
// CHECK-NEXT: |       |     | |-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     | | |-ForStmt [[ADDR_14]] <line:5:3, line:6:5>
// CHECK-NEXT: |       |     | | | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       |     | | | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | | |-<<<NULL>>>
// CHECK-NEXT: |       |     | | | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |     | | | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |     | | | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |     | | | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |     | | | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |     | | | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |     | | `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_28]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_29]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_30]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_31]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |     | `-DeclRefExpr [[ADDR_32]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_33]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_34]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_35]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |     |-RecordDecl [[ADDR_36]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |     | |-CapturedRecordAttr [[ADDR_37]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_38]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_39]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_40]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_41]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | `-FieldDecl [[ADDR_42]] <line:5:3> col:3 implicit 'int'
// CHECK-NEXT: |       |     |   `-OMPCaptureKindAttr [[ADDR_43]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     `-CapturedDecl [[ADDR_13]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |       |-ForStmt [[ADDR_14]] <col:3, line:6:5>
// CHECK-NEXT: |       |       | |-DeclStmt [[ADDR_15]] <line:5:8, col:17>
// CHECK-NEXT: |       |       | | `-VarDecl [[ADDR_16]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | |   `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |-<<<NULL>>>
// CHECK-NEXT: |       |       | |-BinaryOperator [[ADDR_18]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |       | | |-ImplicitCastExpr [[ADDR_19]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | | | `-DeclRefExpr [[ADDR_20]] <col:19> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |       | | `-ImplicitCastExpr [[ADDR_21]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   `-DeclRefExpr [[ADDR_5]] <col:23> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |       |       | |-UnaryOperator [[ADDR_22]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |       | | `-DeclRefExpr [[ADDR_23]] <col:26> 'int' {{.*}}Var [[ADDR_16]] 'i' 'int'
// CHECK-NEXT: |       |       | `-NullStmt [[ADDR_24]] <line:6:5>
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_25]] <line:4:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_26]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_27]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       |       `-VarDecl [[ADDR_16]] <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |         `-IntegerLiteral [[ADDR_17]] <col:16> 'int' 0
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_62:0x[a-z0-9]*]] <col:3> 'int' {{.*}}ParmVar [[ADDR_1]] 'x' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_63:0x[a-z0-9]*]] <line:9:1, line:14:1> line:9:6 test_two 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_64:0x[a-z0-9]*]] <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_65:0x[a-z0-9]*]] <col:22, col:26> col:26 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_66:0x[a-z0-9]*]] <col:29, line:14:1>
// CHECK-NEXT: |   `-OMPTargetTeamsDistributeParallelForDirective [[ADDR_67:0x[a-z0-9]*]] <line:10:1, col:49>
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_68:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr [[ADDR_69:0x[a-z0-9]*]] <line:11:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_70:0x[a-z0-9]*]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_71:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_72:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_73:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_74:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-CapturedStmt [[ADDR_75:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | | | |-CapturedDecl [[ADDR_76:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | |-CapturedStmt [[ADDR_77:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | | | | | |-CapturedDecl [[ADDR_78:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | | |-ForStmt [[ADDR_79:0x[a-z0-9]*]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | | | | | | | |-DeclStmt [[ADDR_80:0x[a-z0-9]*]] <line:11:8, col:17>
// CHECK-NEXT: |       | | | | | | | | | `-VarDecl [[ADDR_81:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | |   `-IntegerLiteral [[ADDR_82:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | | |-BinaryOperator [[ADDR_83:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | | |-ImplicitCastExpr [[ADDR_84:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | | `-DeclRefExpr [[ADDR_85:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | | `-ImplicitCastExpr [[ADDR_86:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | | |-UnaryOperator [[ADDR_87:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_88:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ForStmt [[ADDR_89:0x[a-z0-9]*]] <line:12:5, line:13:7>
// CHECK-NEXT: |       | | | | | | | |   |-DeclStmt [[ADDR_90:0x[a-z0-9]*]] <line:12:10, col:19>
// CHECK-NEXT: |       | | | | | | | |   | `-VarDecl [[ADDR_91:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   |   `-IntegerLiteral [[ADDR_92:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |   |-BinaryOperator [[ADDR_93:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | | |   | |-ImplicitCastExpr [[ADDR_94:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   | | `-DeclRefExpr [[ADDR_95:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | |   | `-ImplicitCastExpr [[ADDR_96:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | | | | | | |   |-UnaryOperator [[ADDR_97:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | |   | `-DeclRefExpr [[ADDR_98:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | |   `-NullStmt [[ADDR_99:0x[a-z0-9]*]] <line:13:7>
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_100:0x[a-z0-9]*]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_101:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_102:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | | | | | | |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_103:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_104:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_105:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_106:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_107:0x[a-z0-9]*]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | `-DeclRefExpr [[ADDR_108:0x[a-z0-9]*]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_109:0x[a-z0-9]*]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_110:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_111:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | | | | |-RecordDecl [[ADDR_112:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | | |-CapturedRecordAttr [[ADDR_113:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_114:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_115:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_116:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_117:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_118:0x[a-z0-9]*]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_119:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | `-FieldDecl [[ADDR_120:0x[a-z0-9]*]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       | | | | | |   `-OMPCaptureKindAttr [[ADDR_121:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | `-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |   |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | | | |   | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       | | | | |   | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | |   | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | | | |   | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       | | | | |   |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       | | | | |   |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | |   |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | | | |   |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | | | |   |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | |     `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | |-DeclRefExpr [[ADDR_122:0x[a-z0-9]*]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_123:0x[a-z0-9]*]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_124:0x[a-z0-9]*]] <line:10:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_125:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_126:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl [[ADDR_127:0x[a-z0-9]*]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | `-OMPCaptureKindAttr [[ADDR_128:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_129:0x[a-z0-9]*]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       | | | |   `-OMPCaptureKindAttr [[ADDR_130:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | `-CapturedDecl [[ADDR_76]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   |-CapturedStmt [[ADDR_77]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | |   | |-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   | | |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | |   | | | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       | | |   | | | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |   | | | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | |   | | | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       | | |   | | |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       | | |   | | |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |   | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | |   | | |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | |   | | |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | |   | | |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |   | |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_103]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_104]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_105]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_106]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_107]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | |   | `-DeclRefExpr [[ADDR_108]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_109]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_110]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_111]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | |   |-RecordDecl [[ADDR_112]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |   | |-CapturedRecordAttr [[ADDR_113]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_114]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_115]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_116]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_117]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_118]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_119]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | `-FieldDecl [[ADDR_120]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       | | |   |   `-OMPCaptureKindAttr [[ADDR_121]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   `-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |     |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       | | |     | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       | | |     | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |-<<<NULL>>>
// CHECK-NEXT: |       | | |     | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |     | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | |     | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | |     | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |     | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       | | |     | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       | | |     |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       | | |     |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |     |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |     |   |-<<<NULL>>>
// CHECK-NEXT: |       | | |     |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | |     |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | |     |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | | |     |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | |     |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       | | |     |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | |     |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |       `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |-DeclRefExpr [[ADDR_131:0x[a-z0-9]*]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_132:0x[a-z0-9]*]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_133:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_134:0x[a-z0-9]*]] <line:10:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_135:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_136:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_137:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_138:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_139:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_140:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_141:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl [[ADDR_142:0x[a-z0-9]*]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr [[ADDR_143:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_144:0x[a-z0-9]*]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_145:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_74]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-CapturedStmt [[ADDR_75]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |   | |-CapturedDecl [[ADDR_76]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | |-CapturedStmt [[ADDR_77]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |   | | | |-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | | |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |   | | | | | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       |   | | | | | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       |   | | | | |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       |   | | | | |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | | |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |   | | | | |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |   | | | | |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_103]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_104]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_105]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_106]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_107]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |   | | | `-DeclRefExpr [[ADDR_108]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_109]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_110]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_111]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |   | | |-RecordDecl [[ADDR_112]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | | |-CapturedRecordAttr [[ADDR_113]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_114]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_115]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_116]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_117]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_118]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_119]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | `-FieldDecl [[ADDR_120]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       |   | | |   `-OMPCaptureKindAttr [[ADDR_121]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | `-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |   |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |   | |   | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       |   | |   | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | |   | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |   | |   | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       |   | |   |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       |   | |   |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | |   |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | |   |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |   | |   |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |   | |   |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |   | |   |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |   | |   |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | |     `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | |-DeclRefExpr [[ADDR_122]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |   | `-DeclRefExpr [[ADDR_123]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_124]] <line:10:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_125]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_126]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl [[ADDR_127]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | `-OMPCaptureKindAttr [[ADDR_128]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_129]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       |   |   `-OMPCaptureKindAttr [[ADDR_130]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   `-CapturedDecl [[ADDR_76]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     |-CapturedStmt [[ADDR_77]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |     | |-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     | | |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |     | | | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       |     | | | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | | |-<<<NULL>>>
// CHECK-NEXT: |       |     | | | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |     | | | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |     | | | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |     | | | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |     | | | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |     | | | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       |     | | |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       |     | | |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |     | | |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |     | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |     | | |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |     | | |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |     | | |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |     | | |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |     | | |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |     | | |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |     | | |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |     | |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_103]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_104]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_105]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_106]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_107]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |     | `-DeclRefExpr [[ADDR_108]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_109]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_110]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_111]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |     |-RecordDecl [[ADDR_112]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |     | |-CapturedRecordAttr [[ADDR_113]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_114]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_115]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_116]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_117]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_118]] <line:11:3> col:3 implicit 'int'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_119]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | `-FieldDecl [[ADDR_120]] <line:12:25> col:25 implicit 'int'
// CHECK-NEXT: |       |     |   `-OMPCaptureKindAttr [[ADDR_121]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     `-CapturedDecl [[ADDR_78]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |       |-ForStmt [[ADDR_79]] <line:11:3, line:13:7>
// CHECK-NEXT: |       |       | |-DeclStmt [[ADDR_80]] <line:11:8, col:17>
// CHECK-NEXT: |       |       | | `-VarDecl [[ADDR_81]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | |   `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |-<<<NULL>>>
// CHECK-NEXT: |       |       | |-BinaryOperator [[ADDR_83]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |       | | |-ImplicitCastExpr [[ADDR_84]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | | | `-DeclRefExpr [[ADDR_85]] <col:19> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |       | | `-ImplicitCastExpr [[ADDR_86]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   `-DeclRefExpr [[ADDR_69]] <col:23> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       |       | |-UnaryOperator [[ADDR_87]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |       | | `-DeclRefExpr [[ADDR_88]] <col:26> 'int' {{.*}}Var [[ADDR_81]] 'i' 'int'
// CHECK-NEXT: |       |       | `-ForStmt [[ADDR_89]] <line:12:5, line:13:7>
// CHECK-NEXT: |       |       |   |-DeclStmt [[ADDR_90]] <line:12:10, col:19>
// CHECK-NEXT: |       |       |   | `-VarDecl [[ADDR_91]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |       |   |   `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |       |   |-<<<NULL>>>
// CHECK-NEXT: |       |       |   |-BinaryOperator [[ADDR_93]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |       |   | |-ImplicitCastExpr [[ADDR_94]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |   | | `-DeclRefExpr [[ADDR_95]] <col:21> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |       |   | `-ImplicitCastExpr [[ADDR_96]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |   |   `-DeclRefExpr [[ADDR_70]] <col:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |       |       |   |-UnaryOperator [[ADDR_97]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |       |   | `-DeclRefExpr [[ADDR_98]] <col:28> 'int' {{.*}}Var [[ADDR_91]] 'i' 'int'
// CHECK-NEXT: |       |       |   `-NullStmt [[ADDR_99]] <line:13:7>
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_100]] <line:10:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_101]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_102]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       |       |-VarDecl [[ADDR_81]] <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | `-IntegerLiteral [[ADDR_82]] <col:16> 'int' 0
// CHECK-NEXT: |       |       `-VarDecl [[ADDR_91]] <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |         `-IntegerLiteral [[ADDR_92]] <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_146:0x[a-z0-9]*]] <line:11:3> 'int' {{.*}}ParmVar [[ADDR_64]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_147:0x[a-z0-9]*]] <line:12:25> 'int' {{.*}}ParmVar [[ADDR_65]] 'y' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_148:0x[a-z0-9]*]] <line:16:1, line:21:1> line:16:6 test_three 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_149:0x[a-z0-9]*]] <col:17, col:21> col:21 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_150:0x[a-z0-9]*]] <col:24, col:28> col:28 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_151:0x[a-z0-9]*]] <col:31, line:21:1>
// CHECK-NEXT: |   `-OMPTargetTeamsDistributeParallelForDirective [[ADDR_152:0x[a-z0-9]*]] <line:17:1, col:61>
// CHECK-NEXT: |     |-OMPCollapseClause [[ADDR_153:0x[a-z0-9]*]] <col:50, col:60>
// CHECK-NEXT: |     | `-ConstantExpr [[ADDR_154:0x[a-z0-9]*]] <col:59> 'int'
// CHECK-NEXT: |     |   |-value: Int 1
// CHECK-NEXT: |     |   `-IntegerLiteral [[ADDR_155:0x[a-z0-9]*]] <col:59> 'int' 1
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_156:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr [[ADDR_157:0x[a-z0-9]*]] <line:18:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_158:0x[a-z0-9]*]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_159:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_160:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_161:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_162:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-CapturedStmt [[ADDR_163:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | | | |-CapturedDecl [[ADDR_164:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | |-CapturedStmt [[ADDR_165:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | | | | | |-CapturedDecl [[ADDR_166:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | | |-ForStmt [[ADDR_167:0x[a-z0-9]*]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | | | | | | | |-DeclStmt [[ADDR_168:0x[a-z0-9]*]] <line:18:8, col:17>
// CHECK-NEXT: |       | | | | | | | | | `-VarDecl [[ADDR_169:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | |   `-IntegerLiteral [[ADDR_170:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | | |-BinaryOperator [[ADDR_171:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | | |-ImplicitCastExpr [[ADDR_172:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | | `-DeclRefExpr [[ADDR_173:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | | `-ImplicitCastExpr [[ADDR_174:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | | |-UnaryOperator [[ADDR_175:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_176:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ForStmt [[ADDR_177:0x[a-z0-9]*]] <line:19:5, line:20:7>
// CHECK-NEXT: |       | | | | | | | |   |-DeclStmt [[ADDR_178:0x[a-z0-9]*]] <line:19:10, col:19>
// CHECK-NEXT: |       | | | | | | | |   | `-VarDecl [[ADDR_179:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   |   `-IntegerLiteral [[ADDR_180:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |   |-BinaryOperator [[ADDR_181:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | | |   | |-ImplicitCastExpr [[ADDR_182:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   | | `-DeclRefExpr [[ADDR_183:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | |   | `-ImplicitCastExpr [[ADDR_184:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | | | | | | |   |-UnaryOperator [[ADDR_185:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | |   | `-DeclRefExpr [[ADDR_186:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | |   `-NullStmt [[ADDR_187:0x[a-z0-9]*]] <line:20:7>
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_188:0x[a-z0-9]*]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_189:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_190:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | | | | | | |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_191:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_192:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_193:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_194:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_195:0x[a-z0-9]*]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | `-DeclRefExpr [[ADDR_196:0x[a-z0-9]*]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_197:0x[a-z0-9]*]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_198:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_199:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | | | | |-RecordDecl [[ADDR_200:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | | |-CapturedRecordAttr [[ADDR_201:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_202:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_203:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_204:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_205:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_206:0x[a-z0-9]*]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_207:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | `-FieldDecl [[ADDR_208:0x[a-z0-9]*]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       | | | | | |   `-OMPCaptureKindAttr [[ADDR_209:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | `-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |   |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | | | |   | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       | | | | |   | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | |   | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | | | |   | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       | | | | |   |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       | | | | |   |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | |   |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | | | |   |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | | | |   |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | |     `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | |-DeclRefExpr [[ADDR_210:0x[a-z0-9]*]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_211:0x[a-z0-9]*]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_212:0x[a-z0-9]*]] <line:17:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_213:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_214:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl [[ADDR_215:0x[a-z0-9]*]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | `-OMPCaptureKindAttr [[ADDR_216:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_217:0x[a-z0-9]*]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       | | | |   `-OMPCaptureKindAttr [[ADDR_218:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | `-CapturedDecl [[ADDR_164]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   |-CapturedStmt [[ADDR_165]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | |   | |-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   | | |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | |   | | | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       | | |   | | | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |   | | | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | |   | | | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       | | |   | | |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       | | |   | | |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |   | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | |   | | |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | |   | | |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | |   | | |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |   | |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_191]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_192]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_193]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_194]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_195]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | |   | `-DeclRefExpr [[ADDR_196]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_197]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_198]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_199]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | |   |-RecordDecl [[ADDR_200]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |   | |-CapturedRecordAttr [[ADDR_201]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_202]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_203]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_204]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_205]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_206]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_207]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | `-FieldDecl [[ADDR_208]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       | | |   |   `-OMPCaptureKindAttr [[ADDR_209]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   `-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |     |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       | | |     | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       | | |     | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |-<<<NULL>>>
// CHECK-NEXT: |       | | |     | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |     | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | |     | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | |     | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |     | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       | | |     | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       | | |     |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       | | |     |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |     |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |     |   |-<<<NULL>>>
// CHECK-NEXT: |       | | |     |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | |     |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | |     |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | | |     |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | |     |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       | | |     |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | |     |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |       `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |-DeclRefExpr [[ADDR_219:0x[a-z0-9]*]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_220:0x[a-z0-9]*]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_221:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_222:0x[a-z0-9]*]] <line:17:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_223:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_224:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_225:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_226:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_227:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_228:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_229:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl [[ADDR_230:0x[a-z0-9]*]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr [[ADDR_231:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_232:0x[a-z0-9]*]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_233:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_162]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-CapturedStmt [[ADDR_163]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |   | |-CapturedDecl [[ADDR_164]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | |-CapturedStmt [[ADDR_165]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |   | | | |-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | | |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |   | | | | | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       |   | | | | | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       |   | | | | |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       |   | | | | |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | | |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |   | | | | |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |   | | | | |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_191]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_192]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_193]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_194]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_195]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |   | | | `-DeclRefExpr [[ADDR_196]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_197]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_198]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_199]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |   | | |-RecordDecl [[ADDR_200]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | | |-CapturedRecordAttr [[ADDR_201]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_202]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_203]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_204]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_205]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_206]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_207]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | `-FieldDecl [[ADDR_208]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       |   | | |   `-OMPCaptureKindAttr [[ADDR_209]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | `-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |   |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |   | |   | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       |   | |   | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | |   | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |   | |   | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       |   | |   |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       |   | |   |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | |   |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | |   |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |   | |   |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |   | |   |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |   | |   |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |   | |   |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | |     `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | |-DeclRefExpr [[ADDR_210]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |   | `-DeclRefExpr [[ADDR_211]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_212]] <line:17:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_213]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_214]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl [[ADDR_215]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | `-OMPCaptureKindAttr [[ADDR_216]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_217]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       |   |   `-OMPCaptureKindAttr [[ADDR_218]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   `-CapturedDecl [[ADDR_164]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     |-CapturedStmt [[ADDR_165]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |     | |-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     | | |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |     | | | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       |     | | | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | | |-<<<NULL>>>
// CHECK-NEXT: |       |     | | | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |     | | | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |     | | | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |     | | | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |     | | | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |     | | | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       |     | | |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       |     | | |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |     | | |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |     | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |     | | |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |     | | |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |     | | |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |     | | |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |     | | |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |     | | |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |     | | |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |     | |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_191]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_192]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_193]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_194]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_195]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |     | `-DeclRefExpr [[ADDR_196]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_197]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_198]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_199]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |     |-RecordDecl [[ADDR_200]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |     | |-CapturedRecordAttr [[ADDR_201]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_202]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_203]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_204]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_205]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_206]] <line:18:3> col:3 implicit 'int'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_207]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | `-FieldDecl [[ADDR_208]] <line:19:25> col:25 implicit 'int'
// CHECK-NEXT: |       |     |   `-OMPCaptureKindAttr [[ADDR_209]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     `-CapturedDecl [[ADDR_166]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |       |-ForStmt [[ADDR_167]] <line:18:3, line:20:7>
// CHECK-NEXT: |       |       | |-DeclStmt [[ADDR_168]] <line:18:8, col:17>
// CHECK-NEXT: |       |       | | `-VarDecl [[ADDR_169]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | |   `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |-<<<NULL>>>
// CHECK-NEXT: |       |       | |-BinaryOperator [[ADDR_171]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |       | | |-ImplicitCastExpr [[ADDR_172]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | | | `-DeclRefExpr [[ADDR_173]] <col:19> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |       | | `-ImplicitCastExpr [[ADDR_174]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   `-DeclRefExpr [[ADDR_157]] <col:23> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       |       | |-UnaryOperator [[ADDR_175]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |       | | `-DeclRefExpr [[ADDR_176]] <col:26> 'int' {{.*}}Var [[ADDR_169]] 'i' 'int'
// CHECK-NEXT: |       |       | `-ForStmt [[ADDR_177]] <line:19:5, line:20:7>
// CHECK-NEXT: |       |       |   |-DeclStmt [[ADDR_178]] <line:19:10, col:19>
// CHECK-NEXT: |       |       |   | `-VarDecl [[ADDR_179]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |       |   |   `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |       |   |-<<<NULL>>>
// CHECK-NEXT: |       |       |   |-BinaryOperator [[ADDR_181]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |       |   | |-ImplicitCastExpr [[ADDR_182]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |   | | `-DeclRefExpr [[ADDR_183]] <col:21> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |       |   | `-ImplicitCastExpr [[ADDR_184]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |   |   `-DeclRefExpr [[ADDR_158]] <col:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |       |       |   |-UnaryOperator [[ADDR_185]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |       |   | `-DeclRefExpr [[ADDR_186]] <col:28> 'int' {{.*}}Var [[ADDR_179]] 'i' 'int'
// CHECK-NEXT: |       |       |   `-NullStmt [[ADDR_187]] <line:20:7>
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_188]] <line:17:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_189]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_190]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       |       |-VarDecl [[ADDR_169]] <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | `-IntegerLiteral [[ADDR_170]] <col:16> 'int' 0
// CHECK-NEXT: |       |       `-VarDecl [[ADDR_179]] <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |         `-IntegerLiteral [[ADDR_180]] <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_234:0x[a-z0-9]*]] <line:18:3> 'int' {{.*}}ParmVar [[ADDR_149]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_235:0x[a-z0-9]*]] <line:19:25> 'int' {{.*}}ParmVar [[ADDR_150]] 'y' 'int'
// CHECK-NEXT: |-FunctionDecl [[ADDR_236:0x[a-z0-9]*]] <line:23:1, line:28:1> line:23:6 test_four 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_237:0x[a-z0-9]*]] <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl [[ADDR_238:0x[a-z0-9]*]] <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_239:0x[a-z0-9]*]] <col:30, line:28:1>
// CHECK-NEXT: |   `-OMPTargetTeamsDistributeParallelForDirective [[ADDR_240:0x[a-z0-9]*]] <line:24:1, col:61>
// CHECK-NEXT: |     |-OMPCollapseClause [[ADDR_241:0x[a-z0-9]*]] <col:50, col:60>
// CHECK-NEXT: |     | `-ConstantExpr [[ADDR_242:0x[a-z0-9]*]] <col:59> 'int'
// CHECK-NEXT: |     |   |-value: Int 2
// CHECK-NEXT: |     |   `-IntegerLiteral [[ADDR_243:0x[a-z0-9]*]] <col:59> 'int' 2
// CHECK-NEXT: |     |-OMPFirstprivateClause [[ADDR_244:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT: |     | |-DeclRefExpr [[ADDR_245:0x[a-z0-9]*]] <line:25:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |     | `-DeclRefExpr [[ADDR_246:0x[a-z0-9]*]] <line:26:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |     `-CapturedStmt [[ADDR_247:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |-CapturedDecl [[ADDR_248:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-CapturedStmt [[ADDR_249:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | |-CapturedDecl [[ADDR_250:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | |-CapturedStmt [[ADDR_251:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | | | |-CapturedDecl [[ADDR_252:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | |-CapturedStmt [[ADDR_253:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | | | | | |-CapturedDecl [[ADDR_254:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | | | | |-ForStmt [[ADDR_255:0x[a-z0-9]*]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | | | | | | | |-DeclStmt [[ADDR_256:0x[a-z0-9]*]] <line:25:8, col:17>
// CHECK-NEXT: |       | | | | | | | | | `-VarDecl [[ADDR_257:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | |   `-IntegerLiteral [[ADDR_258:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | | |-BinaryOperator [[ADDR_259:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | | | | | | |-ImplicitCastExpr [[ADDR_260:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | | | `-DeclRefExpr [[ADDR_261:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | | `-ImplicitCastExpr [[ADDR_262:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | | | |-UnaryOperator [[ADDR_263:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | | | `-DeclRefExpr [[ADDR_264:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | | `-ForStmt [[ADDR_265:0x[a-z0-9]*]] <line:26:5, line:27:7>
// CHECK-NEXT: |       | | | | | | | |   |-DeclStmt [[ADDR_266:0x[a-z0-9]*]] <line:26:10, col:19>
// CHECK-NEXT: |       | | | | | | | |   | `-VarDecl [[ADDR_267:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | |   |   `-IntegerLiteral [[ADDR_268:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | | | | |   |-BinaryOperator [[ADDR_269:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | | | | |   | |-ImplicitCastExpr [[ADDR_270:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   | | `-DeclRefExpr [[ADDR_271:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | |   | `-ImplicitCastExpr [[ADDR_272:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | | | | |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | | | | | | |   |-UnaryOperator [[ADDR_273:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | | | | |   | `-DeclRefExpr [[ADDR_274:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | | | | | | |   `-NullStmt [[ADDR_275:0x[a-z0-9]*]] <line:27:7>
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_276:0x[a-z0-9]*]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_277:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | | | |-ImplicitParamDecl [[ADDR_278:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | | | | | | |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | | | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | | | | `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | | | |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_279:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_280:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_281:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_282:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | | | | | |-DeclRefExpr [[ADDR_283:0x[a-z0-9]*]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | | | | | `-DeclRefExpr [[ADDR_284:0x[a-z0-9]*]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_285:0x[a-z0-9]*]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_286:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | | |-ImplicitParamDecl [[ADDR_287:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | | | | |-RecordDecl [[ADDR_288:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | | | |-CapturedRecordAttr [[ADDR_289:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_290:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_291:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_292:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_293:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | |-FieldDecl [[ADDR_294:0x[a-z0-9]*]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | | | `-OMPCaptureKindAttr [[ADDR_295:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | | `-FieldDecl [[ADDR_296:0x[a-z0-9]*]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       | | | | | |   `-OMPCaptureKindAttr [[ADDR_297:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | | `-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | | | |   |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | | | |   | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       | | | | |   | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   | |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | | |   | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | | | |   | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       | | | | |   |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       | | | | |   |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       | | | | |   |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | | | |   |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | |   |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | | | |   |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | | | |   |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | | | |   |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | | | |   |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | | | |   |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | | | |   | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | | | |   `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | | | |     `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | | | |-DeclRefExpr [[ADDR_298:0x[a-z0-9]*]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | | | `-DeclRefExpr [[ADDR_299:0x[a-z0-9]*]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | | |-ImplicitParamDecl [[ADDR_300:0x[a-z0-9]*]] <line:24:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | | |-RecordDecl [[ADDR_301:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | | | |-CapturedRecordAttr [[ADDR_302:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | | | |-FieldDecl [[ADDR_303:0x[a-z0-9]*]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | | | `-OMPCaptureKindAttr [[ADDR_304:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | | `-FieldDecl [[ADDR_305:0x[a-z0-9]*]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       | | | |   `-OMPCaptureKindAttr [[ADDR_306:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | | `-CapturedDecl [[ADDR_252]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   |-CapturedStmt [[ADDR_253]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | |   | |-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |   | | |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | |   | | | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       | | |   | | | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |   | | | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | |   | | | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       | | |   | | |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       | | |   | | |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |   | | |   |-<<<NULL>>>
// CHECK-NEXT: |       | | |   | | |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | |   | | |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   | | |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | |   | | |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | |   | | |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | |   | | |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   | | |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | |   | | |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   | | | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |   | | `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |   | |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_279]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_280]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_281]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_282]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       | | |   | |-DeclRefExpr [[ADDR_283]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | |   | `-DeclRefExpr [[ADDR_284]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_285]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_286]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |   |-ImplicitParamDecl [[ADDR_287]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | |   |-RecordDecl [[ADDR_288]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |   | |-CapturedRecordAttr [[ADDR_289]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_290]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_291]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_292]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_293]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | |-FieldDecl [[ADDR_294]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | |   | | `-OMPCaptureKindAttr [[ADDR_295]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   | `-FieldDecl [[ADDR_296]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       | | |   |   `-OMPCaptureKindAttr [[ADDR_297]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | |   `-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | | |     |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       | | |     | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       | | |     | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     | |-<<<NULL>>>
// CHECK-NEXT: |       | | |     | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | |     | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | |     | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | |     | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | |     | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       | | |     | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       | | |     |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       | | |     |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |     |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |     |   |-<<<NULL>>>
// CHECK-NEXT: |       | | |     |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | | |     |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | |     |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |     |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | | |     |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | | |     |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       | | |     |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |     |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | |     |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |     | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       | | |     `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | | |       `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       | | |-DeclRefExpr [[ADDR_307:0x[a-z0-9]*]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       | | `-DeclRefExpr [[ADDR_308:0x[a-z0-9]*]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       | |-AlwaysInlineAttr [[ADDR_309:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_310:0x[a-z0-9]*]] <line:24:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_311:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_312:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_313:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_314:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT: |       | |-ImplicitParamDecl [[ADDR_315:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | |-RecordDecl [[ADDR_316:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       | | |-CapturedRecordAttr [[ADDR_317:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       | | |-FieldDecl [[ADDR_318:0x[a-z0-9]*]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       | | | `-OMPCaptureKindAttr [[ADDR_319:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | | `-FieldDecl [[ADDR_320:0x[a-z0-9]*]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       | |   `-OMPCaptureKindAttr [[ADDR_321:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       | `-CapturedDecl [[ADDR_250]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   |-CapturedStmt [[ADDR_251]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |   | |-CapturedDecl [[ADDR_252]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | |-CapturedStmt [[ADDR_253]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |   | | | |-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | | | | |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |   | | | | | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       |   | | | | | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | | |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | | | | | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |   | | | | | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       |   | | | | |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       |   | | | | |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | | | | |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | | | | |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | | | | |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |   | | | | |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | | | | |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |   | | | | |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | | | |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |   | | | | |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | | | | | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | | | | `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | | | |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_279]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_280]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_281]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_282]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |   | | | |-DeclRefExpr [[ADDR_283]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |   | | | `-DeclRefExpr [[ADDR_284]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_285]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_286]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | | |-ImplicitParamDecl [[ADDR_287]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |   | | |-RecordDecl [[ADDR_288]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | | | |-CapturedRecordAttr [[ADDR_289]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_290]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_291]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_292]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_293]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | |-FieldDecl [[ADDR_294]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | | | `-OMPCaptureKindAttr [[ADDR_295]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | | `-FieldDecl [[ADDR_296]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       |   | | |   `-OMPCaptureKindAttr [[ADDR_297]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | | `-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |   | |   |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |   | |   | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       |   | |   | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   | |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |   | |   | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |   | |   | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |   | |   | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       |   | |   |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       |   | |   |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | |   |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | |   |   |-<<<NULL>>>
// CHECK-NEXT: |       |   | |   |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |   | |   |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |   | |   |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   | |   |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |   | |   |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |   | |   |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |   | |   |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |   | |   |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |   | |   |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   | |   | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |   | |   `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   | |     `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |   | |-DeclRefExpr [[ADDR_298]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |   | `-DeclRefExpr [[ADDR_299]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |   |-ImplicitParamDecl [[ADDR_300]] <line:24:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |   |-RecordDecl [[ADDR_301]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |   | |-CapturedRecordAttr [[ADDR_302]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |   | |-FieldDecl [[ADDR_303]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       |   | | `-OMPCaptureKindAttr [[ADDR_304]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   | `-FieldDecl [[ADDR_305]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       |   |   `-OMPCaptureKindAttr [[ADDR_306]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |   `-CapturedDecl [[ADDR_252]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     |-CapturedStmt [[ADDR_253]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |     | |-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |     | | |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |     | | | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       |     | | | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | | |-<<<NULL>>>
// CHECK-NEXT: |       |     | | | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |     | | | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |     | | | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |     | | | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |     | | | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |     | | | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       |     | | |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       |     | | |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |     | | |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |     | | |   |-<<<NULL>>>
// CHECK-NEXT: |       |     | | |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |     | | |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |     | | |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |     | | |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |     | | |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |     | | |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |     | | |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     | | |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |     | | |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |     | | | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |     | | `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |     | |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_279]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_280]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_281]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_282]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT: |       |     | |-DeclRefExpr [[ADDR_283]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |     | `-DeclRefExpr [[ADDR_284]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_285]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_286]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |     |-ImplicitParamDecl [[ADDR_287]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |     |-RecordDecl [[ADDR_288]] <col:1> col:1 implicit struct definition
// CHECK-NEXT: |       |     | |-CapturedRecordAttr [[ADDR_289]] <<invalid sloc>> Implicit
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_290]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_291]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_292]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_293]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | |-FieldDecl [[ADDR_294]] <line:25:3> col:3 implicit 'int'
// CHECK-NEXT: |       |     | | `-OMPCaptureKindAttr [[ADDR_295]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     | `-FieldDecl [[ADDR_296]] <line:26:5> col:5 implicit 'int'
// CHECK-NEXT: |       |     |   `-OMPCaptureKindAttr [[ADDR_297]] <<invalid sloc>> Implicit 30
// CHECK-NEXT: |       |     `-CapturedDecl [[ADDR_254]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       |       |-ForStmt [[ADDR_255]] <line:25:3, line:27:7>
// CHECK-NEXT: |       |       | |-DeclStmt [[ADDR_256]] <line:25:8, col:17>
// CHECK-NEXT: |       |       | | `-VarDecl [[ADDR_257]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | |   `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |       | |-<<<NULL>>>
// CHECK-NEXT: |       |       | |-BinaryOperator [[ADDR_259]] <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       |       | | |-ImplicitCastExpr [[ADDR_260]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | | | `-DeclRefExpr [[ADDR_261]] <col:19> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |       | | `-ImplicitCastExpr [[ADDR_262]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       | |   `-DeclRefExpr [[ADDR_245]] <col:23> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       |       | |-UnaryOperator [[ADDR_263]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       |       | | `-DeclRefExpr [[ADDR_264]] <col:26> 'int' {{.*}}Var [[ADDR_257]] 'i' 'int'
// CHECK-NEXT: |       |       | `-ForStmt [[ADDR_265]] <line:26:5, line:27:7>
// CHECK-NEXT: |       |       |   |-DeclStmt [[ADDR_266]] <line:26:10, col:19>
// CHECK-NEXT: |       |       |   | `-VarDecl [[ADDR_267]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |       |   |   `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |       |   |-<<<NULL>>>
// CHECK-NEXT: |       |       |   |-BinaryOperator [[ADDR_269]] <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       |       |   | |-ImplicitCastExpr [[ADDR_270]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |   | | `-DeclRefExpr [[ADDR_271]] <col:21> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |       |   | `-ImplicitCastExpr [[ADDR_272]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       |       |   |   `-DeclRefExpr [[ADDR_246]] <col:25> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: |       |       |   |-UnaryOperator [[ADDR_273]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       |       |   | `-DeclRefExpr [[ADDR_274]] <col:28> 'int' {{.*}}Var [[ADDR_267]] 'i' 'int'
// CHECK-NEXT: |       |       |   `-NullStmt [[ADDR_275]] <line:27:7>
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_276]] <line:24:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_277]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       |       |-ImplicitParamDecl [[ADDR_278]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       |       |-VarDecl [[ADDR_257]] <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |       | `-IntegerLiteral [[ADDR_258]] <col:16> 'int' 0
// CHECK-NEXT: |       |       `-VarDecl [[ADDR_267]] <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |         `-IntegerLiteral [[ADDR_268]] <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr [[ADDR_322:0x[a-z0-9]*]] <line:25:3> 'int' {{.*}}ParmVar [[ADDR_237]] 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr [[ADDR_323:0x[a-z0-9]*]] <line:26:5> 'int' {{.*}}ParmVar [[ADDR_238]] 'y' 'int'
// CHECK-NEXT: `-FunctionDecl [[ADDR_324:0x[a-z0-9]*]] <line:30:1, line:36:1> line:30:6 test_five 'void (int, int, int)'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_325:0x[a-z0-9]*]] <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_326:0x[a-z0-9]*]] <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT:   |-ParmVarDecl [[ADDR_327:0x[a-z0-9]*]] <col:30, col:34> col:34 used z 'int'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_328:0x[a-z0-9]*]] <col:37, line:36:1>
// CHECK-NEXT:     `-OMPTargetTeamsDistributeParallelForDirective [[ADDR_329:0x[a-z0-9]*]] <line:31:1, col:61>
// CHECK-NEXT:       |-OMPCollapseClause [[ADDR_330:0x[a-z0-9]*]] <col:50, col:60>
// CHECK-NEXT:       | `-ConstantExpr [[ADDR_331:0x[a-z0-9]*]] <col:59> 'int'
// CHECK-NEXT:       |   |-value: Int 2
// CHECK-NEXT:       |   `-IntegerLiteral [[ADDR_332:0x[a-z0-9]*]] <col:59> 'int' 2
// CHECK-NEXT:       |-OMPFirstprivateClause [[ADDR_333:0x[a-z0-9]*]] <<invalid sloc>> <implicit>
// CHECK-NEXT:       | |-DeclRefExpr [[ADDR_334:0x[a-z0-9]*]] <line:32:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:       | |-DeclRefExpr [[ADDR_335:0x[a-z0-9]*]] <line:33:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:       | `-DeclRefExpr [[ADDR_336:0x[a-z0-9]*]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:       `-CapturedStmt [[ADDR_337:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         |-CapturedDecl [[ADDR_338:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | |-CapturedStmt [[ADDR_339:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | |-CapturedDecl [[ADDR_340:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | |-CapturedStmt [[ADDR_341:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | | | |-CapturedDecl [[ADDR_342:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | | |-CapturedStmt [[ADDR_343:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | | | | | |-CapturedDecl [[ADDR_344:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | | | | |-ForStmt [[ADDR_345:0x[a-z0-9]*]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | | | | | | | |-DeclStmt [[ADDR_346:0x[a-z0-9]*]] <line:32:8, col:17>
// CHECK-NEXT:         | | | | | | | | | `-VarDecl [[ADDR_347:0x[a-z0-9]*]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | | |   `-IntegerLiteral [[ADDR_348:0x[a-z0-9]*]] <col:16> 'int' 0
// CHECK-NEXT:         | | | | | | | | |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | | | |-BinaryOperator [[ADDR_349:0x[a-z0-9]*]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | | | | | | | |-ImplicitCastExpr [[ADDR_350:0x[a-z0-9]*]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | | | | `-DeclRefExpr [[ADDR_351:0x[a-z0-9]*]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | | | `-ImplicitCastExpr [[ADDR_352:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | | | | | | | |-UnaryOperator [[ADDR_353:0x[a-z0-9]*]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | | | | `-DeclRefExpr [[ADDR_354:0x[a-z0-9]*]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | | `-ForStmt [[ADDR_355:0x[a-z0-9]*]] <line:33:5, line:35:9>
// CHECK-NEXT:         | | | | | | | |   |-DeclStmt [[ADDR_356:0x[a-z0-9]*]] <line:33:10, col:19>
// CHECK-NEXT:         | | | | | | | |   | `-VarDecl [[ADDR_357:0x[a-z0-9]*]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | |   |   `-IntegerLiteral [[ADDR_358:0x[a-z0-9]*]] <col:18> 'int' 0
// CHECK-NEXT:         | | | | | | | |   |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | | |   |-BinaryOperator [[ADDR_359:0x[a-z0-9]*]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | | | | | |   | |-ImplicitCastExpr [[ADDR_360:0x[a-z0-9]*]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | |   | | `-DeclRefExpr [[ADDR_361:0x[a-z0-9]*]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | |   | `-ImplicitCastExpr [[ADDR_362:0x[a-z0-9]*]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | | | | | | |   |-UnaryOperator [[ADDR_363:0x[a-z0-9]*]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | | |   | `-DeclRefExpr [[ADDR_364:0x[a-z0-9]*]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | |   `-ForStmt [[ADDR_365:0x[a-z0-9]*]] <line:34:7, line:35:9>
// CHECK-NEXT:         | | | | | | | |     |-DeclStmt [[ADDR_366:0x[a-z0-9]*]] <line:34:12, col:21>
// CHECK-NEXT:         | | | | | | | |     | `-VarDecl [[ADDR_367:0x[a-z0-9]*]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | |     |   `-IntegerLiteral [[ADDR_368:0x[a-z0-9]*]] <col:20> 'int' 0
// CHECK-NEXT:         | | | | | | | |     |-<<<NULL>>>
// CHECK-NEXT:         | | | | | | | |     |-BinaryOperator [[ADDR_369:0x[a-z0-9]*]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | | | | | |     | |-ImplicitCastExpr [[ADDR_370:0x[a-z0-9]*]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | |     | | `-DeclRefExpr [[ADDR_371:0x[a-z0-9]*]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | |     | `-ImplicitCastExpr [[ADDR_372:0x[a-z0-9]*]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | | | | |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | | | | | | |     |-UnaryOperator [[ADDR_373:0x[a-z0-9]*]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | | | | | |     | `-DeclRefExpr [[ADDR_374:0x[a-z0-9]*]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | | | | | | |     `-NullStmt [[ADDR_375:0x[a-z0-9]*]] <line:35:9>
// CHECK-NEXT:         | | | | | | | |-ImplicitParamDecl [[ADDR_376:0x[a-z0-9]*]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | | | |-ImplicitParamDecl [[ADDR_377:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | | | |-ImplicitParamDecl [[ADDR_378:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | | | | | | |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | | | | | | |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | | | | | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | | | | | | `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | | | |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | | | | | |-DeclRefExpr [[ADDR_379:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_380:0x[a-z0-9]*]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         | | | | | | |-DeclRefExpr [[ADDR_381:0x[a-z0-9]*]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_382:0x[a-z0-9]*]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         | | | | | | |-DeclRefExpr [[ADDR_383:0x[a-z0-9]*]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | | | | | |-DeclRefExpr [[ADDR_384:0x[a-z0-9]*]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | | | | | `-DeclRefExpr [[ADDR_385:0x[a-z0-9]*]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | | | | |-ImplicitParamDecl [[ADDR_386:0x[a-z0-9]*]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | |-ImplicitParamDecl [[ADDR_387:0x[a-z0-9]*]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | | |-ImplicitParamDecl [[ADDR_388:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | | | | |-RecordDecl [[ADDR_389:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | | | | |-CapturedRecordAttr [[ADDR_390:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | | | | |-FieldDecl [[ADDR_391:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         | | | | | | | `-OMPCaptureKindAttr [[ADDR_392:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | | | |-FieldDecl [[ADDR_393:0x[a-z0-9]*]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         | | | | | | | `-OMPCaptureKindAttr [[ADDR_394:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | | | |-FieldDecl [[ADDR_395:0x[a-z0-9]*]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         | | | | | | | `-OMPCaptureKindAttr [[ADDR_396:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | | | |-FieldDecl [[ADDR_397:0x[a-z0-9]*]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         | | | | | | | `-OMPCaptureKindAttr [[ADDR_398:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | | | `-FieldDecl [[ADDR_399:0x[a-z0-9]*]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         | | | | | |   `-OMPCaptureKindAttr [[ADDR_400:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | | `-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | | | |   |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | | | |   | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         | | | | |   | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | |   | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | | | |   | |-<<<NULL>>>
// CHECK-NEXT:         | | | | |   | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | | |   | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | |   | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | | | |   | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | |   | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | | | |   | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | | |   | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | | | |   | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         | | | | |   |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         | | | | |   |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | |   |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | | | |   |   |-<<<NULL>>>
// CHECK-NEXT:         | | | | |   |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | | | |   |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | |   |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | | | |   |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | |   |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | | | |   |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | | | |   |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | | | |   |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         | | | | |   |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         | | | | |   |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | |   |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | | | |   |     |-<<<NULL>>>
// CHECK-NEXT:         | | | | |   |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | | | |   |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | |   |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | | | |   |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | |   |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | | | |   |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | | | |   |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | | | |   |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         | | | | |   |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | |   |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | | | |   |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | | | |   |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | | | |   | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | | | |   |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | | | |   | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | | | |   `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | | | |     `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | | | |-DeclRefExpr [[ADDR_401:0x[a-z0-9]*]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | | | |-DeclRefExpr [[ADDR_402:0x[a-z0-9]*]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | | | `-DeclRefExpr [[ADDR_403:0x[a-z0-9]*]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | | |-ImplicitParamDecl [[ADDR_404:0x[a-z0-9]*]] <line:31:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | | |-RecordDecl [[ADDR_405:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | | | |-CapturedRecordAttr [[ADDR_406:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | | | |-FieldDecl [[ADDR_407:0x[a-z0-9]*]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         | | | | | `-OMPCaptureKindAttr [[ADDR_408:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | |-FieldDecl [[ADDR_409:0x[a-z0-9]*]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         | | | | | `-OMPCaptureKindAttr [[ADDR_410:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | | `-FieldDecl [[ADDR_411:0x[a-z0-9]*]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         | | | |   `-OMPCaptureKindAttr [[ADDR_412:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | | `-CapturedDecl [[ADDR_342]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | |   |-CapturedStmt [[ADDR_343]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | |   | |-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | |   | | |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | |   | | | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         | | |   | | | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |   | | | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | |   | | | |-<<<NULL>>>
// CHECK-NEXT:         | | |   | | | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | |   | | | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   | | | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | |   | | | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   | | | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | |   | | | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | |   | | | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | |   | | | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         | | |   | | |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         | | |   | | |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | |   | | |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | |   | | |   |-<<<NULL>>>
// CHECK-NEXT:         | | |   | | |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | |   | | |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   | | |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | |   | | |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   | | |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | |   | | |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | |   | | |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | |   | | |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         | | |   | | |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         | | |   | | |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | |   | | |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | |   | | |     |-<<<NULL>>>
// CHECK-NEXT:         | | |   | | |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | |   | | |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   | | |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | |   | | |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   | | |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | |   | | |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | |   | | |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | |   | | |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         | | |   | | |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |   | | |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |   | | |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | |   | | |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |   | | | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | |   | | |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | |   | | | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | |   | | `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | |   | |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | |   | |-DeclRefExpr [[ADDR_379]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_380]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         | | |   | |-DeclRefExpr [[ADDR_381]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_382]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         | | |   | |-DeclRefExpr [[ADDR_383]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | |   | |-DeclRefExpr [[ADDR_384]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | |   | `-DeclRefExpr [[ADDR_385]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | |   |-ImplicitParamDecl [[ADDR_386]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |   |-ImplicitParamDecl [[ADDR_387]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |   |-ImplicitParamDecl [[ADDR_388]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | |   |-RecordDecl [[ADDR_389]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | |   | |-CapturedRecordAttr [[ADDR_390]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | |   | |-FieldDecl [[ADDR_391]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         | | |   | | `-OMPCaptureKindAttr [[ADDR_392]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |   | |-FieldDecl [[ADDR_393]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         | | |   | | `-OMPCaptureKindAttr [[ADDR_394]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |   | |-FieldDecl [[ADDR_395]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         | | |   | | `-OMPCaptureKindAttr [[ADDR_396]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |   | |-FieldDecl [[ADDR_397]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         | | |   | | `-OMPCaptureKindAttr [[ADDR_398]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |   | `-FieldDecl [[ADDR_399]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         | | |   |   `-OMPCaptureKindAttr [[ADDR_400]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |   `-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | | |     |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         | | |     | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         | | |     | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |     | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | |     | |-<<<NULL>>>
// CHECK-NEXT:         | | |     | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | |     | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | |     | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | |     | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | |     | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         | | |     | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         | | |     |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         | | |     |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | |     |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | |     |   |-<<<NULL>>>
// CHECK-NEXT:         | | |     |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | | |     |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | |     |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | |     |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | | |     |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         | | |     |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         | | |     |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         | | |     |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | |     |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | |     |     |-<<<NULL>>>
// CHECK-NEXT:         | | |     |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | | |     |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | |     |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |     |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | | |     |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | | |     |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         | | |     |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         | | |     |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |     |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |     |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | |     |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |     | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         | | |     |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | |     | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         | | |     `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | | |       `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         | | |-DeclRefExpr [[ADDR_413:0x[a-z0-9]*]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         | | |-DeclRefExpr [[ADDR_414:0x[a-z0-9]*]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         | | `-DeclRefExpr [[ADDR_415:0x[a-z0-9]*]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         | |-AlwaysInlineAttr [[ADDR_416:0x[a-z0-9]*]] <<invalid sloc>> Implicit __forceinline
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_417:0x[a-z0-9]*]] <line:31:1> col:1 implicit .global_tid. 'const int'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_418:0x[a-z0-9]*]] <col:1> col:1 implicit .part_id. 'const int *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_419:0x[a-z0-9]*]] <col:1> col:1 implicit .privates. 'void *const restrict'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_420:0x[a-z0-9]*]] <col:1> col:1 implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_421:0x[a-z0-9]*]] <col:1> col:1 implicit .task_t. 'void *const'
// CHECK-NEXT:         | |-ImplicitParamDecl [[ADDR_422:0x[a-z0-9]*]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | |-RecordDecl [[ADDR_423:0x[a-z0-9]*]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         | | |-CapturedRecordAttr [[ADDR_424:0x[a-z0-9]*]] <<invalid sloc>> Implicit
// CHECK-NEXT:         | | |-FieldDecl [[ADDR_425:0x[a-z0-9]*]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr [[ADDR_426:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | |-FieldDecl [[ADDR_427:0x[a-z0-9]*]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         | | | `-OMPCaptureKindAttr [[ADDR_428:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | | `-FieldDecl [[ADDR_429:0x[a-z0-9]*]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         | |   `-OMPCaptureKindAttr [[ADDR_430:0x[a-z0-9]*]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         | `-CapturedDecl [[ADDR_340]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   |-CapturedStmt [[ADDR_341]] <line:32:3, line:35:9>
// CHECK-NEXT:         |   | |-CapturedDecl [[ADDR_342]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | | |-CapturedStmt [[ADDR_343]] <line:32:3, line:35:9>
// CHECK-NEXT:         |   | | | |-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | | | | |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         |   | | | | | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         |   | | | | | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | | | | | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |   | | | | | |-<<<NULL>>>
// CHECK-NEXT:         |   | | | | | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   | | | | | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |   | | | | | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |   | | | | | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   | | | | | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |   | | | | | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         |   | | | | |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         |   | | | | |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | | | | |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |   | | | | |   |-<<<NULL>>>
// CHECK-NEXT:         |   | | | | |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   | | | | |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |   | | | | |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |   | | | | |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   | | | | |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |   | | | | |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         |   | | | | |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         |   | | | | |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | | | | |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |   | | | | |     |-<<<NULL>>>
// CHECK-NEXT:         |   | | | | |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   | | | | |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |   | | | | |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | | | | |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |   | | | | |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   | | | | |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |   | | | | |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         |   | | | | |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | | | |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | | | |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |   | | | | |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | | | | | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |   | | | | |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | | | | | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |   | | | | `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | | | |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |   | | | |-DeclRefExpr [[ADDR_379]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_380]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         |   | | | |-DeclRefExpr [[ADDR_381]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_382]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         |   | | | |-DeclRefExpr [[ADDR_383]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |   | | | |-DeclRefExpr [[ADDR_384]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |   | | | `-DeclRefExpr [[ADDR_385]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |   | | |-ImplicitParamDecl [[ADDR_386]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | |-ImplicitParamDecl [[ADDR_387]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | | |-ImplicitParamDecl [[ADDR_388]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |   | | |-RecordDecl [[ADDR_389]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |   | | | |-CapturedRecordAttr [[ADDR_390]] <<invalid sloc>> Implicit
// CHECK-NEXT:         |   | | | |-FieldDecl [[ADDR_391]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         |   | | | | `-OMPCaptureKindAttr [[ADDR_392]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | | | |-FieldDecl [[ADDR_393]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         |   | | | | `-OMPCaptureKindAttr [[ADDR_394]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | | | |-FieldDecl [[ADDR_395]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         |   | | | | `-OMPCaptureKindAttr [[ADDR_396]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | | | |-FieldDecl [[ADDR_397]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         |   | | | | `-OMPCaptureKindAttr [[ADDR_398]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | | | `-FieldDecl [[ADDR_399]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         |   | | |   `-OMPCaptureKindAttr [[ADDR_400]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | | `-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |   | |   |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         |   | |   | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         |   | |   | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | |   | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |   | |   | |-<<<NULL>>>
// CHECK-NEXT:         |   | |   | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |   | |   | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | |   | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |   | |   | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | |   | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |   | |   | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |   | |   | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |   | |   | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         |   | |   |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         |   | |   |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | |   |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |   | |   |   |-<<<NULL>>>
// CHECK-NEXT:         |   | |   |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |   | |   |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | |   |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |   | |   |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | |   |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |   | |   |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |   | |   |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |   | |   |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         |   | |   |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         |   | |   |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | |   |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |   | |   |     |-<<<NULL>>>
// CHECK-NEXT:         |   | |   |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |   | |   |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | |   |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |   | |   |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |   | |   |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |   | |   |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |   | |   |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |   | |   |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         |   | |   |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | |   |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |   | |   |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |   | |   |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   | |   | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |   | |   |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |   | |   | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |   | |   `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   | |     `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |   | |-DeclRefExpr [[ADDR_401]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |   | |-DeclRefExpr [[ADDR_402]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |   | `-DeclRefExpr [[ADDR_403]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |   |-ImplicitParamDecl [[ADDR_404]] <line:31:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |   |-RecordDecl [[ADDR_405]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |   | |-CapturedRecordAttr [[ADDR_406]] <<invalid sloc>> Implicit
// CHECK-NEXT:         |   | |-FieldDecl [[ADDR_407]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         |   | | `-OMPCaptureKindAttr [[ADDR_408]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | |-FieldDecl [[ADDR_409]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         |   | | `-OMPCaptureKindAttr [[ADDR_410]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   | `-FieldDecl [[ADDR_411]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         |   |   `-OMPCaptureKindAttr [[ADDR_412]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |   `-CapturedDecl [[ADDR_342]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |     |-CapturedStmt [[ADDR_343]] <line:32:3, line:35:9>
// CHECK-NEXT:         |     | |-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |     | | |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         |     | | | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         |     | | | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |     | | | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |     | | | |-<<<NULL>>>
// CHECK-NEXT:         |     | | | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |     | | | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |     | | | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |     | | | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |     | | | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |     | | | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |     | | | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |     | | | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         |     | | |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         |     | | |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |     | | |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |     | | |   |-<<<NULL>>>
// CHECK-NEXT:         |     | | |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |     | | |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |     | | |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |     | | |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |     | | |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |     | | |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |     | | |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |     | | |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         |     | | |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         |     | | |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |     | | |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |     | | |     |-<<<NULL>>>
// CHECK-NEXT:         |     | | |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |     | | |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |     | | |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |     | | |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |     | | |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |     | | |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |     | | |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |     | | |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         |     | | |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |     | | |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |     | | |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |     | | |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |     | | | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |     | | |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |     | | | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |     | | `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |     | |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |     | |-DeclRefExpr [[ADDR_379]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_380]] '.captured.omp.previous.lb' 'const unsigned long'
// CHECK-NEXT:         |     | |-DeclRefExpr [[ADDR_381]] <<invalid sloc>> 'const unsigned long' {{.*}}Var [[ADDR_382]] '.captured.omp.previous.ub' 'const unsigned long'
// CHECK-NEXT:         |     | |-DeclRefExpr [[ADDR_383]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |     | |-DeclRefExpr [[ADDR_384]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |     | `-DeclRefExpr [[ADDR_385]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |     |-ImplicitParamDecl [[ADDR_386]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |     |-ImplicitParamDecl [[ADDR_387]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |     |-ImplicitParamDecl [[ADDR_388]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |     |-RecordDecl [[ADDR_389]] <col:1> col:1 implicit struct definition
// CHECK-NEXT:         |     | |-CapturedRecordAttr [[ADDR_390]] <<invalid sloc>> Implicit
// CHECK-NEXT:         |     | |-FieldDecl [[ADDR_391]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         |     | | `-OMPCaptureKindAttr [[ADDR_392]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |     | |-FieldDecl [[ADDR_393]] <<invalid sloc>> <invalid sloc> implicit 'const unsigned long'
// CHECK-NEXT:         |     | | `-OMPCaptureKindAttr [[ADDR_394]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |     | |-FieldDecl [[ADDR_395]] <line:32:3> col:3 implicit 'int'
// CHECK-NEXT:         |     | | `-OMPCaptureKindAttr [[ADDR_396]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |     | |-FieldDecl [[ADDR_397]] <line:33:5> col:5 implicit 'int'
// CHECK-NEXT:         |     | | `-OMPCaptureKindAttr [[ADDR_398]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |     | `-FieldDecl [[ADDR_399]] <line:34:27> col:27 implicit 'int'
// CHECK-NEXT:         |     |   `-OMPCaptureKindAttr [[ADDR_400]] <<invalid sloc>> Implicit 30
// CHECK-NEXT:         |     `-CapturedDecl [[ADDR_344]] <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         |       |-ForStmt [[ADDR_345]] <line:32:3, line:35:9>
// CHECK-NEXT:         |       | |-DeclStmt [[ADDR_346]] <line:32:8, col:17>
// CHECK-NEXT:         |       | | `-VarDecl [[ADDR_347]] <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |       | |   `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |       | |-<<<NULL>>>
// CHECK-NEXT:         |       | |-BinaryOperator [[ADDR_349]] <col:19, col:23> 'int' '<'
// CHECK-NEXT:         |       | | |-ImplicitCastExpr [[ADDR_350]] <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         |       | | | `-DeclRefExpr [[ADDR_351]] <col:19> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |       | | `-ImplicitCastExpr [[ADDR_352]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |       | |   `-DeclRefExpr [[ADDR_334]] <col:23> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |       | |-UnaryOperator [[ADDR_353]] <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         |       | | `-DeclRefExpr [[ADDR_354]] <col:26> 'int' {{.*}}Var [[ADDR_347]] 'i' 'int'
// CHECK-NEXT:         |       | `-ForStmt [[ADDR_355]] <line:33:5, line:35:9>
// CHECK-NEXT:         |       |   |-DeclStmt [[ADDR_356]] <line:33:10, col:19>
// CHECK-NEXT:         |       |   | `-VarDecl [[ADDR_357]] <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |       |   |   `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |       |   |-<<<NULL>>>
// CHECK-NEXT:         |       |   |-BinaryOperator [[ADDR_359]] <col:21, col:25> 'int' '<'
// CHECK-NEXT:         |       |   | |-ImplicitCastExpr [[ADDR_360]] <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         |       |   | | `-DeclRefExpr [[ADDR_361]] <col:21> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |       |   | `-ImplicitCastExpr [[ADDR_362]] <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         |       |   |   `-DeclRefExpr [[ADDR_335]] <col:25> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         |       |   |-UnaryOperator [[ADDR_363]] <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         |       |   | `-DeclRefExpr [[ADDR_364]] <col:28> 'int' {{.*}}Var [[ADDR_357]] 'i' 'int'
// CHECK-NEXT:         |       |   `-ForStmt [[ADDR_365]] <line:34:7, line:35:9>
// CHECK-NEXT:         |       |     |-DeclStmt [[ADDR_366]] <line:34:12, col:21>
// CHECK-NEXT:         |       |     | `-VarDecl [[ADDR_367]] <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |       |     |   `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |       |     |-<<<NULL>>>
// CHECK-NEXT:         |       |     |-BinaryOperator [[ADDR_369]] <col:23, col:27> 'int' '<'
// CHECK-NEXT:         |       |     | |-ImplicitCastExpr [[ADDR_370]] <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         |       |     | | `-DeclRefExpr [[ADDR_371]] <col:23> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |       |     | `-ImplicitCastExpr [[ADDR_372]] <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         |       |     |   `-DeclRefExpr [[ADDR_336]] <col:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
// CHECK-NEXT:         |       |     |-UnaryOperator [[ADDR_373]] <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         |       |     | `-DeclRefExpr [[ADDR_374]] <col:30> 'int' {{.*}}Var [[ADDR_367]] 'i' 'int'
// CHECK-NEXT:         |       |     `-NullStmt [[ADDR_375]] <line:35:9>
// CHECK-NEXT:         |       |-ImplicitParamDecl [[ADDR_376]] <line:31:1> col:1 implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         |       |-ImplicitParamDecl [[ADDR_377]] <col:1> col:1 implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         |       |-ImplicitParamDecl [[ADDR_378]] <col:1> col:1 implicit __context 'struct ({{.*}}ast-dump-openmp-target-teams-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         |       |-VarDecl [[ADDR_347]] <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |       | `-IntegerLiteral [[ADDR_348]] <col:16> 'int' 0
// CHECK-NEXT:         |       |-VarDecl [[ADDR_357]] <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         |       | `-IntegerLiteral [[ADDR_358]] <col:18> 'int' 0
// CHECK-NEXT:         |       `-VarDecl [[ADDR_367]] <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |         `-IntegerLiteral [[ADDR_368]] <col:20> 'int' 0
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_431:0x[a-z0-9]*]] <line:32:3> 'int' {{.*}}ParmVar [[ADDR_325]] 'x' 'int'
// CHECK-NEXT:         |-DeclRefExpr [[ADDR_432:0x[a-z0-9]*]] <line:33:5> 'int' {{.*}}ParmVar [[ADDR_326]] 'y' 'int'
// CHECK-NEXT:         `-DeclRefExpr [[ADDR_433:0x[a-z0-9]*]] <line:34:27> 'int' {{.*}}ParmVar [[ADDR_327]] 'z' 'int'
