// RUN: clang-diff -dump-matches %S/Inputs/clang-diff-basic-src.cpp %s -- | FileCheck %s

// CHECK: Match TranslationUnitDecl{{.*}} to TranslationUnitDecl
// CHECK: Match NamespaceDecl: src{{.*}} to NamespaceDecl: dst
namespace dst {
// CHECK-NOT: Match NamespaceDecl: src{{.*}} to NamespaceDecl: inner
namespace inner {
void foo() {
  // CHECK: Match IntegerLiteral: 321{{.*}} to IntegerLiteral: 322
  int x = 322;
}
}

// CHECK: Match DeclRefExpr: foo{{.*}} to DeclRefExpr: inner::foo
void main() { inner::foo(); }

// CHECK: Match StringLiteral: foo{{.*}} to StringLiteral: foo
const char *b = "f" "o" "o";

// unsigned is canonicalized to unsigned int
// CHECK: Match TypedefDecl: nat;unsigned int;{{.*}} to TypedefDecl: nat;unsigned int;
typedef unsigned nat;

// CHECK: Match VarDecl: p(int){{.*}} to VarDecl: prod(double)
// CHECK: Update VarDecl: p(int){{.*}} to prod(double)
// CHECK: Match BinaryOperator: *{{.*}} to BinaryOperator: *
double prod = 1 * 2 * 10;
// CHECK: Update DeclRefExpr
int squared = prod * prod;

class X {
  const char *foo(int i) {
    if (i == 0)
      return "Bar";
    // CHECK: Insert IfStmt{{.*}} into IfStmt
    // CHECK: Insert BinaryOperator: =={{.*}} into IfStmt
    else if (i == -1)
      return "foo";
    return 0;
  }
  X(){}
};
}

// CHECK: Move CompoundStmt{{.*}} into CompoundStmt
void m() { { int x = 0 + 0 + 0; } }
// CHECK: Update and Move IntegerLiteral: 7{{.*}} into BinaryOperator: +({{.*}}) at 1
int um = 1 + 7;

namespace {
// match with parents of different type
// CHECK: Match FunctionDecl: f1{{.*}} to FunctionDecl: f1
void f1() {{ (void) __func__;;; }}
}

// CHECK: Delete AccessSpecDecl: public
// CHECK: Delete CXXMethodDecl
