// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2b -ast-dump %s | FileCheck %s

union U {
  int a;
  float b;
  U &operator=(this U &self, const U &) = default;
  U &operator=(this U &self, U &&) = default;
};

void odr_use(U &x, const U &y, U &&z) {
  x = y;
  x = static_cast<U &&>(z);
}

// A defaulted union assignment operator written with a C++23 explicit object
// parameter is synthesized with a whole-object __builtin_memcpy body whose
// pointer arguments are cast to void* so -Wnontrivial-memcall stays quiet.

// CHECK: CXXMethodDecl {{.*}} operator= 'U &(U &, const U &)
// CHECK:   CompoundStmt
// CHECK:     CallExpr
// CHECK:       DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK:       CStyleCastExpr {{.*}} 'void *'
// CHECK:       CStyleCastExpr {{.*}} 'const void *'
// CHECK:     ReturnStmt

// CHECK: CXXMethodDecl {{.*}} operator= 'U &(U &, U &&)
// CHECK:   CompoundStmt
// CHECK:     CallExpr
// CHECK:       DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK:       CStyleCastExpr {{.*}} 'void *'
// CHECK:       CStyleCastExpr {{.*}} 'const void *'
// CHECK:     ReturnStmt
