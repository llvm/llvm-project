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
#endif


// CHECK: FunctionDecl {{.*}} TestTaskTransparent 'void ()'
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_not_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: OMPFirstprivateClause
// CHECK-NEXT: DeclRefExpr {{.*}} 'omp_impex_t':'void **' lvalue Var {{.*}} 'imp' 'omp_impex_t':'void **' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'omp_impex_t':'void **' lvalue Var {{.*}} 'imp' 'omp_impex_t':'void **' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: CapturedStmt
// CHECK:  OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_export' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt
// CHECK: OMPTaskLoopDirective
// CHECK-NEXT: OMPTransparentClause
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'omp_impex_t':'void **' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'const omp_impex_t':'void **const' lvalue Var {{.*}} 'omp_impex' 'const omp_impex_t':'void **const'
// CHECK-NEXT: CapturedStmt



