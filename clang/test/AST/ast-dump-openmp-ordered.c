// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(void) {
#pragma omp ordered
  ;
}

void test_two(int x) {
#pragma omp for ordered
  for (int i = 0; i < x; i++)
    ;
}

void test_three(int x) {
#pragma omp for ordered(1)
  for (int i = 0; i < x; i++) {
#pragma omp ordered depend(source)
  }
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: |-FunctionDecl {{.*}} <{{.*}}ast-dump-openmp-ordered.c:3:1, line:6:1> line:3:6 test_one 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:21, line:6:1>
// CHECK-NEXT: |   `-OMPOrderedDirective {{.*}} <line:4:1, col:20>
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:5:3>
// CHECK-NEXT: |       `-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |         |-NullStmt {{.*}} <col:3>
// CHECK-NEXT: |         `-ImplicitParamDecl {{.*}} <line:4:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-ordered.c:4:1) *const restrict'
// CHECK-NEXT: |-FunctionDecl {{.*}} <line:8:1, line:12:1> line:8:6 test_two 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} <col:22, line:12:1>
// CHECK-NEXT: |   `-OMPForDirective {{.*}} <line:9:1, col:24>
// CHECK-NEXT: |     |-OMPOrderedClause {{.*}} <col:17, col:24>
// CHECK-NEXT: |     | `-<<<NULL>>>
// CHECK-NEXT: |     `-CapturedStmt {{.*}} <line:10:3, line:11:5>
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |       | |-ForStmt {{.*}} <line:10:3, line:11:5>
// CHECK-NEXT: |       | | |-DeclStmt {{.*}} <line:10:8, col:17>
// CHECK-NEXT: |       | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-NullStmt {{.*}} <line:11:5>
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} <line:9:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-ordered.c:9:1) *const restrict'
// CHECK-NEXT: |       | `-VarDecl {{.*}} <line:10:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT: |       |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} <col:3> 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: `-FunctionDecl {{.*}} <line:14:1, line:19:1> line:14:6 test_three 'void (int)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:17, col:21> col:21 used x 'int'
// CHECK-NEXT:   `-CompoundStmt {{.*}} <col:24, line:19:1>
// CHECK-NEXT:     `-OMPForDirective {{.*}} <line:15:1, col:27>
// CHECK-NEXT:       |-OMPOrderedClause {{.*}} <col:17, col:26>
// CHECK-NEXT:       | `-ConstantExpr {{.*}} <col:25> 'int'
// CHECK-NEXT:       | |-value: Int 1
// CHECK-NEXT:       |   `-IntegerLiteral {{.*}} <col:25> 'int' 1
// CHECK-NEXT:       `-CapturedStmt {{.*}} <line:16:3, line:18:3>
// CHECK-NEXT:         |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT:         | |-ForStmt {{.*}} <line:16:3, line:18:3>
// CHECK-NEXT:         | | |-DeclStmt {{.*}} <line:16:8, col:17>
// CHECK-NEXT:         | | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         | | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         | | |-<<<NULL>>>
// CHECK-NEXT:         | | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
// CHECK-NEXT:         | | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
// CHECK-NEXT:         | | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | `-CompoundStmt {{.*}} <col:31, line:18:3>
// CHECK-NEXT:         | |   `-OMPOrderedDirective {{.*}} <line:17:1, col:35> openmp_standalone_directive
// CHECK-NEXT:         | |     `-OMPDependClause {{.*}} <col:21, <invalid sloc>>
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} <line:15:1> col:1 implicit __context 'struct (unnamed at {{.*}}ast-dump-openmp-ordered.c:15:1) *const restrict'
// CHECK-NEXT:         | `-VarDecl {{.*}} <line:16:8, col:16> col:12 used i 'int' cinit
// CHECK-NEXT:         |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
// CHECK-NEXT:         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue ParmVar {{.*}} 'x' 'int'
