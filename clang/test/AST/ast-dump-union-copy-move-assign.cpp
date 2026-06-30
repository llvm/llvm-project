// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -ast-dump %s | FileCheck %s

union U {
  int a;
  float b;
};

void odr_use(U &x, const U &y, U &&z) {
  x = y;
  x = static_cast<U &&>(z);
}

// The implicitly-defined defaulted union assignment operators are synthesized
// with a whole-object __builtin_memcpy body whose pointer arguments are cast to
// void* so the copy is not flagged by -Wnontrivial-memcall.

// CHECK: CXXMethodDecl {{.*}} implicit {{.*}}operator= 'U &(const U &)
// CHECK:   CompoundStmt
// CHECK:     CallExpr
// CHECK:       DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK:       CStyleCastExpr {{.*}} 'void *'
// CHECK:       CStyleCastExpr {{.*}} 'const void *'
// CHECK:     ReturnStmt

// CHECK: CXXMethodDecl {{.*}} implicit {{.*}}operator= 'U &(U &&)
// CHECK:   CompoundStmt
// CHECK:     CallExpr
// CHECK:       DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK:       CStyleCastExpr {{.*}} 'void *'
// CHECK:       CStyleCastExpr {{.*}} 'const void *'
// CHECK:     ReturnStmt
