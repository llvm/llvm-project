// RUN: clang-diff -ast-dump %s -- -std=c++11 | FileCheck %s


// CHECK: {{^}}TranslationUnitDecl(0)
// CHECK: {{^}} NamespaceDecl: test;(
namespace test {

// CHECK: {{^}}  FunctionDecl: f(
// CHECK: CompoundStmt(
void f() {
  // CHECK: VarDecl: i(int)(
  // CHECK: IntegerLiteral: 1
  auto i = 1;
  // CHECK: CallExpr(
  // CHECK-NOT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr: f(
  f();
  // CHECK: BinaryOperator: =(
  i = i;
}

} // end namespace test

// CHECK: TypedefDecl: nat;unsigned int;(
typedef unsigned nat;
// CHECK: TypeAliasDecl: real;double;(
using real = double;

class Base {
};

// CHECK: CXXRecordDecl: X;class X;(
class X : Base {
  int m;
  // CHECK: CXXMethodDecl: foo(const char *(int)
  // CHECK: ParmVarDecl: i(int)(
  const char *foo(int i) {
    if (i == 0)
      // CHECK: StringLiteral: foo(
      return "foo";
    // CHECK-NOT: ImplicitCastExpr
    return 0;
  }

  // CHECK: AccessSpecDecl: public(
public:
  // CHECK: CXXConstructorDecl: X(void (char, int){{( __attribute__\(\(thiscall\)\))?}})(
  X(char, int) : Base(), m(0) {
    // CHECK: MemberExpr(
    int x = m;
  }
};

#define M (void)1
#define MA(a, b) (void)a, b
// CHECK: FunctionDecl
// CHECK-NEXT: CompoundStmt
void macros() {
  M;
  MA(1, 2);
}

#ifndef GUARD
#define GUARD
// CHECK-NEXT: NamespaceDecl
namespace world {
// nodes from other files are excluded, there should be no output here
#include "clang-diff-ast.cpp"
}
// CHECK-NEXT: FunctionDecl: sentinel
void sentinel();
#endif
