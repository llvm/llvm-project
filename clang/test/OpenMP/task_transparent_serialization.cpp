// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -include-pch %t -ast-dump-all  %s | FileCheck %s

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

typedef void **omp_impex_t;
extern const omp_impex_t omp_not_impex;
extern const omp_impex_t omp_import;
extern const omp_impex_t omp_export;
extern const omp_impex_t omp_impex;

// CHECK: FunctionDecl {{.*}} TestTaskTransparent 'void ()'
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_not_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'omp_impex_t':'void **' lvalue Var {{.*}} 'imp' 'omp_impex_t':'void **' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'omp_impex_t':'void **' lvalue Var {{.*}} 'imp' 'omp_impex_t':'void **' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt
// CHECK-NEXT: CapturedDecl
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: BinaryOperator {{.*}} 'int' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:30> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: IntegerLiteral {{.*}} <col:32> 'int' 1
// CHECK-NEXT: CapturedStmt
// CHEC-NEXT: CapturedDecl
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT:  OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: BinaryOperator {{.*}} 'int' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: IntegerLiteral {{.*}} <col:32> 'int' 1
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr{{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
/// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
template <typename T>
class TransparentTemplate {
public:
  void TestTaskLoopImpex() {
    #pragma omp taskloop transparent(omp_impex)
    for (int i = 0; i < 10; ++i) {}
  }
};

void TestTaskTransparent() {
  int a;
  omp_impex_t imp;
#pragma omp task transparent(omp_not_impex)
#pragma omp task transparent(imp)
#pragma omp task transparent(a)
#pragma omp task transparent(a+1)

#pragma omp parallel
  {
#pragma omp task transparent(omp_export)
    {
#pragma omp taskloop transparent(omp_impex)
      for (int i = 0; i < 5; ++i) {}
    }
  }
  TransparentTemplate<int> obj;
  obj.TestTaskLoopImpex();
}


// CHECK: FunctionDecl {{.*}} TestTransparentImplicitFirstprivateOnEnclosingTask1 'void ()'

// outer task (1)
// CHECK: OMPTaskDirective

// 'a' is firstprivate on outer
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt

// inner task (2)
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt
void TestTransparentImplicitFirstprivateOnEnclosingTask1() {
  int a;
#pragma omp task                // (1)
  {
#pragma omp task transparent(a) // (2)
    {}
  }
}

// CHECK: FunctionDecl {{.*}} TestTransparentImplicitFirstprivateOnEnclosingTask2 'void ()'

// outer task (1)
// CHECK: OMPTaskDirective

// 'a' is firstprivate on outer task (1)
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt

// inner task (2)
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture

// 'a' is firstprivate on outer task (2)
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt

// inner task (3)
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: BinaryOperator {{.*}} 'int' '+'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: CapturedStmt
void TestTransparentImplicitFirstprivateOnEnclosingTask2() {
  int a;
#pragma omp task                  // (1)
  {
#pragma omp task transparent(a)   // (2)
#pragma omp task transparent(a+1) // (3)
    {}
  }
}

// CHECK: FunctionDecl {{.*}} TestTransparentImplicitFirstprivateOnEnclosingTask3 'void ()'

// outer task (1)
// CHECK: OMPTaskDirective

// 'a' is firstprivate on outer task (1)
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt

// inner task (2)
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'a' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt

// inner task (3)
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
void TestTransparentImplicitFirstprivateOnEnclosingTask3() {
  int a;
#pragma omp task                        // (1)
  {
#pragma omp task transparent(a)         // (2)
#pragma omp task transparent(omp_impex) // (3)
    {}
  }
}

#endif

