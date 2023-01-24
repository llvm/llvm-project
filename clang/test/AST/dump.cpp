// RUN: %clang_cc1 -verify -fopenmp -ast-dump %s | FileCheck %s -implicit-check-not=openmp_structured_block
// RUN: %clang_cc1 -verify -fopenmp-simd -ast-dump %s | FileCheck %s -implicit-check-not=openmp_structured_block
// expected-no-diagnostics

int ga, gb;
#pragma omp threadprivate(ga, gb)

// CHECK:      |-OMPThreadPrivateDecl {{.+}} <col:1> col:1
// CHECK-NEXT: | |-DeclRefExpr {{.+}} <col:27> 'int' lvalue Var {{.+}} 'ga' 'int'
// CHECK-NEXT: | `-DeclRefExpr {{.+}} <col:31> 'int' lvalue Var {{.+}} 'gb' 'int'

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)

#pragma omp declare reduction(fun : float : omp_out += omp_in) initializer(omp_priv = omp_orig + 15)

// CHECK:      |-OMPDeclareReductionDecl {{.+}} <line:[[@LINE-4]]:35> col:35 operator+ 'int' combiner 0x{{.+}}
// CHECK-NEXT: | |-CompoundAssignOperator {{.+}} <col:47, col:58> 'int' lvalue '*=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: | | |-DeclRefExpr {{.+}} <col:47> 'int' lvalue Var {{.+}} 'omp_out' 'int'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:58> 'int' <LValueToRValue>
// CHECK-NEXT: | |   `-DeclRefExpr {{.+}} <col:58> 'int' lvalue Var {{.+}} 'omp_in' 'int'
// CHECK-NEXT: | |-VarDecl {{.+}} <col:35> col:35 implicit used omp_in 'int'
// CHECK-NEXT: | `-VarDecl {{.+}} <col:35> col:35 implicit used omp_out 'int'
// CHECK-NEXT: |-OMPDeclareReductionDecl {{.+}} <col:40> col:40 operator+ 'char' combiner 0x{{.+}}
// CHECK-NEXT: | |-CompoundAssignOperator {{.+}} <col:47, col:58> 'char' lvalue '*=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: | | |-DeclRefExpr {{.+}} <col:47> 'char' lvalue Var {{.+}} 'omp_out' 'char'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:58> 'int' <IntegralCast>
// CHECK-NEXT: | |   `-ImplicitCastExpr {{.+}} <col:58> 'char' <LValueToRValue>
// CHECK-NEXT: | |     `-DeclRefExpr {{.+}} <col:58> 'char' lvalue Var {{.+}} 'omp_in' 'char'
// CHECK-NEXT: | |-VarDecl {{.+}} <col:40> col:40 implicit used omp_in 'char'
// CHECK-NEXT: | `-VarDecl {{.+}} <col:40> col:40 implicit used omp_out 'char'
// CHECK-NEXT: |-OMPDeclareReductionDecl {{.+}} <line:[[@LINE-17]]:37> col:37 fun 'float' combiner 0x{{.+}} initializer 0x{{.+}}
// CHECK-NEXT: | |-CompoundAssignOperator {{.+}} <col:45, col:56> 'float' lvalue '+=' ComputeLHSTy='float' ComputeResultTy='float'
// CHECK-NEXT: | | |-BOOM{{.+}} <col:45> 'float' lvalue Var {{.+}} 'omp_out' 'float'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:56> 'float' <LValueToRValue>
// CHECK-NEXT: | |   `-DeclRefExpr {{.+}} <col:56> 'float' lvalue Var {{.+}} 'omp_in' 'float'
// CHECK-NEXT: | |-BinaryOperator {{.+}} <col:87, col:98> 'float' '+'
// CHECK-NEXT: | | |-ImplicitCastExpr {{.+}} <col:87> 'float' <LValueToRValue>
// CHECK-NEXT: | | | `-DeclRefExpr {{.+}} <col:87> 'float' lvalue Var {{.+}} 'omp_orig' 'float'
// CHECK-NEXT: | | `-ImplicitCastExpr {{.+}} <col:98> 'float' <IntegralToFloating>
// CHECK-NEXT: | |   `-IntegerLiteral {{.+}} <col:98> 'int' 15

struct S {
  int a, b;
  S() {
#pragma omp parallel for default(none) private(a) shared(b) schedule(static, a)
    for (int i = 0; i < 0; ++i)
      ++a;
  }
};

// CHECK:      |     `-OMPParallelForDirective {{.+}} {{<line:.+:1, col:80>|<col:1, col:80>}}
// CHECK-NEXT: |       |-OMPDefaultClause {{.+}} <col:26, col:38>
// CHECK-NEXT: |       |-OMPPrivateClause {{.+}} <col:40, col:49>
// CHECK-NEXT: |       | `-DeclRefExpr {{.+}} <col:48> 'int' lvalue OMPCapturedExpr {{.+}} 'a' 'int &'
// CHECK-NEXT: |       |-OMPSharedClause {{.+}} <col:51, col:59>
// CHECK-NEXT: |       | `-MemberExpr {{.+}} <col:58> 'int' lvalue ->b
// CHECK-NEXT: |       |   `-CXXThisExpr {{.+}} <col:58> 'S *' implicit this
// CHECK-NEXT: |       |-OMPScheduleClause {{.+}} <col:61, col:79>
// CHECK-NEXT: |       | `-ImplicitCastExpr {{.+}} <col:78> 'int' <LValueToRValue>
// CHECK-NEXT: |       |   `-DeclRefExpr {{.+}} <col:78> 'int' lvalue OMPCapturedExpr {{.+}} '.capture_expr.' 'int'
// CHECK-NEXT: |       `-CapturedStmt {{.+}} <line:[[@LINE-15]]:5, line:[[@LINE-14]]:9>
// CHECK-NEXT: |         |-CapturedDecl {{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |         | |-ForStmt {{.+}} <line:[[@LINE-17]]:5, line:[[@LINE-16]]:9>
// CHECK:      |         | | `-UnaryOperator {{.+}} <line:[[@LINE-17]]:7, col:9> 'int' lvalue prefix '++'
// CHECK-NEXT: |         | |   `-DeclRefExpr {{.+}} <col:9> 'int' lvalue OMPCapturedExpr {{.+}} 'a' 'int &'

#pragma omp declare simd
#pragma omp declare simd inbranch
void foo();

// CHECK:        |-FunctionDecl {{.+}} <line:[[@LINE-2]]:1, col:10> col:6 foo 'void ()'
// CHECK-NEXT:   |-OMPDeclareSimdDeclAttr {{.+}} <line:[[@LINE-4]]:1, col:34> Implicit BS_Inbranch
// CHECK:        `-OMPDeclareSimdDeclAttr {{.+}} <line:[[@LINE-6]]:1, col:25> Implicit BS_Undefined

#pragma omp declare target
int bar() {
  int f;
  return f;
}
#pragma omp end declare target

// CHECK:       `-FunctionDecl {{.+}} <line:[[@LINE-6]]:1, line:[[@LINE-3]]:1> line:[[@LINE-6]]:5 bar 'int ()'
// CHECK-NEXT:  |-CompoundStmt {{.+}} <col:11, line:[[@LINE-4]]:1>
// CHECK-NEXT:  | |-DeclStmt {{.+}} <line:[[@LINE-7]]:3, col:8>
// CHECK-NEXT:  | | `-VarDecl {{.+}} <col:3, col:7> col:7 used f 'int'
// CHECK-NEXT:  | `-ReturnStmt {{.+}} <line:[[@LINE-8]]:3, col:10>
// CHECK-NEXT:  |   `-ImplicitCastExpr {{.+}} <col:10> 'int' <LValueToRValue>
// CHECK-NEXT:  |     `-DeclRefExpr {{.+}} <col:10> 'int' lvalue Var {{.+}} 'f' 'int'
// CHECK-NEXT:  `-OMPDeclareTargetDeclAttr {{.+}} <line:75:21> Implicit MT_To DT_Any 1
