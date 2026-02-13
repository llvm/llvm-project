// RUN: %clang_cc1 -ast-dump %s 2>&1 | FileCheck %s

// CHECK-LABEL: example0
void example0() {
  double d = 2.0;
  // CHECK: VarDecl{{.*}}rd 'double &'
  // CHECK-NEXT: DeclRefExpr
  double &rd = d;
  // CHECK: VarDecl{{.*}}rcd 'const double &'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'const double' lvalue <NoOp>
  const double &rcd = d;
}

struct A { };
struct B : A { } b;

// CHECK-LABEL: example1
void example1() {
  // CHECK: VarDecl{{.*}}ra 'A &'
  // CHECK: ImplicitCastExpr{{.*}}'A' lvalue <DerivedToBase (A)>
  A &ra = b;
  // CHECK: VarDecl{{.*}}rca 'const A &'
  // CHECK: ImplicitCastExpr{{.*}}'const A' lvalue <DerivedToBase (A)>
  // CHECK-NOT: MaterializeTemporaryExpr
  // CHECK: ImplicitCastExpr{{.*}}'const struct B' lvalue <NoOp>
  const A& rca = b;
}

extern B f();

struct X {
  operator B();
} x;

// CHECK-LABEL: example2
void example2() {
  // CHECK: VarDecl{{.*}}rca 'const A &'
  // CHECK: ImplicitCastExpr{{.*}}'const A' lvalue <DerivedToBase (A)>
  // CHECK: MaterializeTemporaryExpr{{.*}}'const B'
  // CHECK: ImplicitCastExpr{{.*}}'const B' <NoOp>
  // CHECK: CallExpr{{.*}}B
  const A &rca = f();
  // CHECK: VarDecl{{.*}}r 'const A &'
  // CHECK: ImplicitCastExpr{{.*}}'const A' lvalue <DerivedToBase (A)>
  // CHECK: MaterializeTemporaryExpr{{.*}}'const B'
  // CHECK: ImplicitCastExpr{{.*}}'const B' <NoOp>
  // CHECK: CXXMemberCallExpr{{.*}}'B'
  const A& r = x;
}

// CHECK-LABEL: example3
void example3() {
  // CHECK: VarDecl{{.*}}rcd2 'const double &'
  // CHECK: ImplicitCastExpr{{.*}} <IntegralToFloating>
  const double& rcd2 = 2;
}
