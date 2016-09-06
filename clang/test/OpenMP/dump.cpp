// RUN: %clang_cc1 -verify -fopenmp -ast-dump %s | FileCheck %s
// expected-no-diagnostics

int ga, gb;
#pragma omp threadprivate(ga, gb)

// CHECK:      |-OMPThreadPrivateDecl {{.+}} <col:9> col:9
// CHECK-NEXT: | |-DeclRefExpr {{.+}} <col:27> 'int' lvalue Var {{.+}} 'ga' 'int'
// CHECK-NEXT: | `-DeclRefExpr {{.+}} <col:31> 'int' lvalue Var {{.+}} 'gb' 'int'

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)

#pragma omp declare reduction(fun : float : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)

// CHECK:      |-OMPDeclareReductionDecl {{.+}} <line:11:35> col:35 operator+ 'int' combiner
// CHECK-NEXT: | |-CompoundAssignOperator {{.+}} <col:47, col:58> 'int' lvalue '*=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: | | |-DeclRefExpr {{.+}} <col:47> 'int' lvalue Var {{.+}} 'omp_out' 'int'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:58> 'int' <LValueToRValue>
// CHECK-NEXT: | |   `-DeclRefExpr {{.+}} <col:58> 'int' lvalue Var {{.+}} 'omp_in' 'int'
// CHECK-NEXT: | |-VarDecl {{.+}} <col:35> col:35 implicit used omp_in 'int'
// CHECK-NEXT: | `-VarDecl {{.+}} <col:35> col:35 implicit used omp_out 'int'
// CHECK-NEXT: |-OMPDeclareReductionDecl {{.+}} <col:40> col:40 operator+ 'char' combiner
// CHECK-NEXT: | |-CompoundAssignOperator {{.+}} <col:47, col:58> 'char' lvalue '*=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: | | |-DeclRefExpr {{.+}} <col:47> 'char' lvalue Var {{.+}} 'omp_out' 'char'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:58> 'int' <IntegralCast>
// CHECK-NEXT: | |   `-ImplicitCastExpr {{.+}} <col:58> 'char' <LValueToRValue>
// CHECK-NEXT: | |     `-DeclRefExpr {{.+}} <col:58> 'char' lvalue Var {{.+}} 'omp_in' 'char'
// CHECK-NEXT: | |-VarDecl {{.+}} <col:40> col:40 implicit used omp_in 'char'
// CHECK-NEXT: | `-VarDecl {{.+}} <col:40> col:40 implicit used omp_out 'char'
// CHECK-NEXT: |-OMPDeclareReductionDecl {{.+}} <line:13:37> col:37 fun 'float' combiner initializer
// CHECK-NEXT: | |-CompoundAssignOperator {{.+}} <col:45, col:56> 'float' lvalue '+=' ComputeLHSTy='float' ComputeResultTy='float'
// CHECK-NEXT: | | |-DeclRefExpr {{.+}} <col:45> 'float' lvalue Var {{.+}} 'omp_out' 'float'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:56> 'float' <LValueToRValue>
// CHECK-NEXT: | |   `-DeclRefExpr {{.+}} <col:56> 'float' lvalue Var {{.+}} 'omp_in' 'float'

struct S {
  int a, b;
  S() {
#pragma omp parallel for default(none) private(a) shared(b) schedule(static, a)
    for (int i = 0; i < 0; ++i)
      ++a;
  }
};

// CHECK:      |     `-OMPParallelForDirective {{.+}} <line:39:9, col:80>
// CHECK-NEXT: |       |-OMPDefaultClause {{.+}} <col:26, col:40>
// CHECK-NEXT: |       |-OMPPrivateClause {{.+}} <col:40, col:51>
// CHECK-NEXT: |       | `-DeclRefExpr {{.+}} <col:48> 'int' lvalue OMPCapturedExpr {{.+}} 'a' 'int &'
// CHECK-NEXT: |       |-OMPSharedClause {{.+}} <col:51, col:61>
// CHECK-NEXT: |       | `-MemberExpr {{.+}} <col:58> 'int' lvalue ->b
// CHECK-NEXT: |       |   `-CXXThisExpr {{.+}} <col:58> 'struct S *' this
// CHECK-NEXT: |       |-OMPScheduleClause {{.+}} <col:61, col:79>
// CHECK-NEXT: |       | `-ImplicitCastExpr {{.+}} <col:78> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   `-DeclRefExpr {{.+}} <col:78> 'int' lvalue OMPCapturedExpr {{.+}} '.capture_expr.' 'int'
// CHECK-NEXT: |       |-CapturedStmt {{.+}} <line:40:5, <invalid sloc>>
// CHECK-NEXT: |       | |-CapturedDecl {{.+}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |       | | |-ForStmt {{.+}} <col:5, <invalid sloc>>
// CHECK:      |       | | | `-UnaryOperator {{.+}} <line:41:7, <invalid sloc>> 'int' lvalue prefix '++'
// CHECK-NEXT: |       | | |   `-DeclRefExpr {{.+}} <<invalid sloc>> 'int' lvalue OMPCapturedExpr {{.+}} 'a' 'int &'

#pragma omp declare simd
#pragma omp declare simd inbranch
void foo();

// CHECK:      `-FunctionDecl {{.+}} <line:63:1, col:10> col:6 foo 'void (void)'
// CHECK-NEXT:   |-OMPDeclareSimdDeclAttr {{.+}} <line:62:9, col:34> Implicit BS_Inbranch
// CHECK:        `-OMPDeclareSimdDeclAttr {{.+}} <line:61:9, col:25> Implicit BS_Undefined

