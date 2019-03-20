// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(int x) {
#pragma omp for
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
#pragma omp for
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
#pragma omp for collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
#pragma omp for collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
#pragma omp for collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}

// CHECK: TranslationUnitDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: |-FunctionDecl 0x{{.*}} <{{.*}}ast-dump-openmp-for.c:3:1, line:7:1> line:3:6 test_one 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | `-CompoundStmt 0x{{.*}} <col:22, line:7:1>
// CHECK-NEXT: |   `-OMPForDirective 0x{{.*}} <line:4:9, col:16>
// CHECK-NEXT: |     `-CapturedStmt 0x{{.*}} <line:5:3, line:6:5>
// CHECK-NEXT: |       |-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |       | |-ForStmt 0x{{.*}} <line:5:3, line:6:5>
// CHECK-NEXT: |       | | |-DeclStmt 0x{{.*}} <line:5:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl 0x{{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator 0x{{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr 0x{{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr 0x{{.*}} <col:19> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr 0x{{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr 0x{{.*}} <col:23> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator 0x{{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr 0x{{.*}} <col:26> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-NullStmt 0x{{.*}} <line:6:5> openmp_structured_block
// CHECK-NEXT: |       | |-ImplicitParamDecl 0x{{.*}} <line:4:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-for.c:4:9) *const restrict'
// CHECK-NEXT: |       | `-VarDecl 0x{{.*}} <line:5:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       `-DeclRefExpr 0x{{.*}} <col:3> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |-FunctionDecl 0x{{.*}} <line:9:1, line:14:1> line:9:6 test_two 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:22, col:26> col:26 used y 'int'
// CHECK-NEXT: | `-CompoundStmt 0x{{.*}} <col:29, line:14:1>
// CHECK-NEXT: |   `-OMPForDirective 0x{{.*}} <line:10:9, col:16>
// CHECK-NEXT: |     `-CapturedStmt 0x{{.*}} <line:11:3, line:13:7>
// CHECK-NEXT: |       |-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |       | |-ForStmt 0x{{.*}} <line:11:3, line:13:7>
// CHECK-NEXT: |       | | |-DeclStmt 0x{{.*}} <line:11:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl 0x{{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator 0x{{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr 0x{{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr 0x{{.*}} <col:19> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr 0x{{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr 0x{{.*}} <col:23> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator 0x{{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr 0x{{.*}} <col:26> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt 0x{{.*}} <line:12:5, line:13:7> openmp_structured_block
// CHECK-NEXT: |       | |   |-DeclStmt 0x{{.*}} <line:12:10, col:19>
// CHECK-NEXT: |       | |   | `-VarDecl 0x{{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | |   |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator 0x{{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr 0x{{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr 0x{{.*}} <col:21> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr 0x{{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr 0x{{.*}} <col:25> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT: |       | |   |-UnaryOperator 0x{{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr 0x{{.*}} <col:28> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt 0x{{.*}} <line:13:7>
// CHECK-NEXT: |       | |-ImplicitParamDecl 0x{{.*}} <line:10:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-for.c:10:9) *const restrict'
// CHECK-NEXT: |       | |-VarDecl 0x{{.*}} <line:11:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | `-VarDecl 0x{{.*}} <line:12:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr 0x{{.*}} <line:11:3> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr 0x{{.*}} <line:12:25> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT: |-FunctionDecl 0x{{.*}} <line:16:1, line:21:1> line:16:6 test_three 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:17, col:21> col:21 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:24, col:28> col:28 used y 'int'
// CHECK-NEXT: | `-CompoundStmt 0x{{.*}} <col:31, line:21:1>
// CHECK-NEXT: |   `-OMPForDirective 0x{{.*}} <line:17:9, col:28>
// CHECK-NEXT: |     |-OMPCollapseClause 0x{{.*}} <col:17, col:27>
// CHECK-NEXT: |     | `-ConstantExpr 0x{{.*}} <col:26> 'int'
// CHECK-NEXT: |     |   `-IntegerLiteral 0x{{.*}} <col:26> 'int' 1
// CHECK-NEXT: |     `-CapturedStmt 0x{{.*}} <line:18:3, line:20:7>
// CHECK-NEXT: |       |-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |       | |-ForStmt 0x{{.*}} <line:18:3, line:20:7>
// CHECK-NEXT: |       | | |-DeclStmt 0x{{.*}} <line:18:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl 0x{{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator 0x{{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr 0x{{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr 0x{{.*}} <col:19> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr 0x{{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr 0x{{.*}} <col:23> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator 0x{{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr 0x{{.*}} <col:26> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt 0x{{.*}} <line:19:5, line:20:7> openmp_structured_block
// CHECK-NEXT: |       | |   |-DeclStmt 0x{{.*}} <line:19:10, col:19>
// CHECK-NEXT: |       | |   | `-VarDecl 0x{{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | |   |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator 0x{{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr 0x{{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr 0x{{.*}} <col:21> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr 0x{{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr 0x{{.*}} <col:25> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT: |       | |   |-UnaryOperator 0x{{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr 0x{{.*}} <col:28> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt 0x{{.*}} <line:20:7>
// CHECK-NEXT: |       | |-ImplicitParamDecl 0x{{.*}} <line:17:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-for.c:17:9) *const restrict'
// CHECK-NEXT: |       | |-VarDecl 0x{{.*}} <line:18:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | `-VarDecl 0x{{.*}} <line:19:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr 0x{{.*}} <line:18:3> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr 0x{{.*}} <line:19:25> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT: |-FunctionDecl 0x{{.*}} <line:23:1, line:28:1> line:23:6 test_four 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT: | |-ParmVarDecl 0x{{.*}} <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT: | `-CompoundStmt 0x{{.*}} <col:30, line:28:1>
// CHECK-NEXT: |   `-OMPForDirective 0x{{.*}} <line:24:9, col:28>
// CHECK-NEXT: |     |-OMPCollapseClause 0x{{.*}} <col:17, col:27>
// CHECK-NEXT: |     | `-ConstantExpr 0x{{.*}} <col:26> 'int'
// CHECK-NEXT: |     |   `-IntegerLiteral 0x{{.*}} <col:26> 'int' 2
// CHECK-NEXT: |     `-CapturedStmt 0x{{.*}} <line:25:3, line:27:7>
// CHECK-NEXT: |       |-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |       | |-ForStmt 0x{{.*}} <line:25:3, line:27:7>
// CHECK-NEXT: |       | | |-DeclStmt 0x{{.*}} <line:25:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl 0x{{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator 0x{{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr 0x{{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr 0x{{.*}} <col:19> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr 0x{{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr 0x{{.*}} <col:23> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       | | |-UnaryOperator 0x{{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr 0x{{.*}} <col:26> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt 0x{{.*}} <line:26:5, line:27:7>
// CHECK-NEXT: |       | |   |-DeclStmt 0x{{.*}} <line:26:10, col:19>
// CHECK-NEXT: |       | |   | `-VarDecl 0x{{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       | |   |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator 0x{{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr 0x{{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr 0x{{.*}} <col:21> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr 0x{{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr 0x{{.*}} <col:25> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT: |       | |   |-UnaryOperator 0x{{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr 0x{{.*}} <col:28> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt 0x{{.*}} <line:27:7> openmp_structured_block
// CHECK-NEXT: |       | |-ImplicitParamDecl 0x{{.*}} <line:24:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-for.c:24:9) *const restrict'
// CHECK-NEXT: |       | |-VarDecl 0x{{.*}} <line:25:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | `-VarDecl 0x{{.*}} <line:26:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT: |       |-DeclRefExpr 0x{{.*}} <line:25:3> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr 0x{{.*}} <line:26:5> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT: `-FunctionDecl 0x{{.*}} <line:30:1, line:36:1> line:30:6 test_five 'void (int, int, int)'
// CHECK-NEXT:   |-ParmVarDecl 0x{{.*}} <col:16, col:20> col:20 used x 'int'
// CHECK-NEXT:   |-ParmVarDecl 0x{{.*}} <col:23, col:27> col:27 used y 'int'
// CHECK-NEXT:   |-ParmVarDecl 0x{{.*}} <col:30, col:34> col:34 used z 'int'
// CHECK-NEXT:   `-CompoundStmt 0x{{.*}} <col:37, line:36:1>
// CHECK-NEXT:     `-OMPForDirective 0x{{.*}} <line:31:9, col:28>
// CHECK-NEXT:       |-OMPCollapseClause 0x{{.*}} <col:17, col:27>
// CHECK-NEXT:       | `-ConstantExpr 0x{{.*}} <col:26> 'int'
// CHECK-NEXT:       |   `-IntegerLiteral 0x{{.*}} <col:26> 'int' 2
// CHECK-NEXT:       `-CapturedStmt 0x{{.*}} <line:32:3, line:35:9>
// CHECK-NEXT:         |-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:         | |-ForStmt 0x{{.*}} <line:32:3, line:35:9>
// CHECK-NEXT:         | | |-DeclStmt 0x{{.*}} <line:32:8, col:17>
// CHECK-NEXT:         | | | `-VarDecl 0x{{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |   `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | |-<<<NULL>>>
// CHECK-NEXT:         | | |-BinaryOperator 0x{{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |-ImplicitCastExpr 0x{{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | `-DeclRefExpr 0x{{.*}} <col:19> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT:         | | | `-ImplicitCastExpr 0x{{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   `-DeclRefExpr 0x{{.*}} <col:23> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT:         | | |-UnaryOperator 0x{{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | `-DeclRefExpr 0x{{.*}} <col:26> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT:         | | `-ForStmt 0x{{.*}} <line:33:5, line:35:9>
// CHECK-NEXT:         | |   |-DeclStmt 0x{{.*}} <line:33:10, col:19>
// CHECK-NEXT:         | |   | `-VarDecl 0x{{.*}} <col:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | |   |   `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | |   |-<<<NULL>>>
// CHECK-NEXT:         | |   |-BinaryOperator 0x{{.*}} <col:21, col:25> 'int' '<'
// CHECK-NEXT:         | |   | |-ImplicitCastExpr 0x{{.*}} <col:21> 'int' <LValueToRValue>
// CHECK-NEXT:         | |   | | `-DeclRefExpr 0x{{.*}} <col:21> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT:         | |   | `-ImplicitCastExpr 0x{{.*}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:         | |   |   `-DeclRefExpr 0x{{.*}} <col:25> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT:         | |   |-UnaryOperator 0x{{.*}} <col:28, col:29> 'int' postfix '++'
// CHECK-NEXT:         | |   | `-DeclRefExpr 0x{{.*}} <col:28> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT:         | |   `-ForStmt 0x{{.*}} <line:34:7, line:35:9> openmp_structured_block
// CHECK-NEXT:         | |     |-DeclStmt 0x{{.*}} <line:34:12, col:21>
// CHECK-NEXT:         | |     | `-VarDecl 0x{{.*}} <col:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         | |     |   `-IntegerLiteral 0x{{.*}} <col:20> 'int' 0
// CHECK-NEXT:         | |     |-<<<NULL>>>
// CHECK-NEXT:         | |     |-BinaryOperator 0x{{.*}} <col:23, col:27> 'int' '<'
// CHECK-NEXT:         | |     | |-ImplicitCastExpr 0x{{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | |     | | `-DeclRefExpr 0x{{.*}} <col:23> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT:         | |     | `-ImplicitCastExpr 0x{{.*}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT:         | |     |   `-DeclRefExpr 0x{{.*}} <col:27> 'int' lvalue ParmVar 0x{{.*}} 'z' 'int'
// CHECK-NEXT:         | |     |-UnaryOperator 0x{{.*}} <col:30, col:31> 'int' postfix '++'
// CHECK-NEXT:         | |     | `-DeclRefExpr 0x{{.*}} <col:30> 'int' lvalue Var 0x{{.*}} 'i' 'int'
// CHECK-NEXT:         | |     `-NullStmt 0x{{.*}} <line:35:9>
// CHECK-NEXT:         | |-ImplicitParamDecl 0x{{.*}} <line:31:9> col:9 implicit __context 'struct (anonymous at {{.*}}ast-dump-openmp-for.c:31:9) *const restrict'
// CHECK-NEXT:         | |-VarDecl 0x{{.*}} <line:32:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | `-IntegerLiteral 0x{{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | |-VarDecl 0x{{.*}} <line:33:10, col:18> col:14 used i 'int' cinit
// CHECK-NEXT:         | | `-IntegerLiteral 0x{{.*}} <col:18> 'int' 0
// CHECK-NEXT:         | `-VarDecl 0x{{.*}} <line:34:12, col:20> col:16 used i 'int' cinit
// CHECK-NEXT:         |   `-IntegerLiteral 0x{{.*}} <col:20> 'int' 0
// CHECK-NEXT:         |-DeclRefExpr 0x{{.*}} <line:32:3> 'int' lvalue ParmVar 0x{{.*}} 'x' 'int'
// CHECK-NEXT:         |-DeclRefExpr 0x{{.*}} <line:33:5> 'int' lvalue ParmVar 0x{{.*}} 'y' 'int'
// CHECK-NEXT:         `-DeclRefExpr 0x{{.*}} <line:34:27> 'int' lvalue ParmVar 0x{{.*}} 'z' 'int'
