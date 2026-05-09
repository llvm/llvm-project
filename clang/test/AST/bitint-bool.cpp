// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

typedef unsigned _BitInt(1) b1;

void test(b1 x) {
  if (x) {
    int a = 1;
  }
}

// CHECK: FunctionDecl
// CHECK: IfStmt
// CHECK: ImplicitCastExpr