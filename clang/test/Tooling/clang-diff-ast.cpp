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
  // CHECK: DeclRefExpr: f(
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
  // CHECK: CXXMethodDecl: foo(const char *(int))(
  // CHECK: ParmVarDecl: i(int)(
  const char *foo(int i) {
    if (i == 0)
      // CHECK: StringLiteral: foo(
      return "foo";
    return 0;
  }

  // CHECK: AccessSpecDecl: public(
public:
  // CHECK: CXXConstructorDecl: X(void (char, int))(
  X(char, int) : Base(), m(0) {
    // CHECK: MemberExpr(
    int x = m;
  }
};
