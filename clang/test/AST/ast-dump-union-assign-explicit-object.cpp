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

// C++23 explicit-object form uses the same typed-pointer memcpy.

// CHECK: CXXMethodDecl {{.*}} operator= 'U &(U &, const U &)
// CHECK:   CompoundStmt
// CHECK:     CallExpr
// CHECK:       DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK:       ImplicitCastExpr {{.*}} 'void *' <BitCast>
// CHECK:         UnaryOperator {{.*}} 'U *' prefix '&'
// CHECK:       ImplicitCastExpr {{.*}} 'const void *' <BitCast>
// CHECK:         UnaryOperator {{.*}} 'const U *' prefix '&'
// CHECK:     ReturnStmt

// CHECK: CXXMethodDecl {{.*}} operator= 'U &(U &, U &&)
// CHECK:   CompoundStmt
// CHECK:     CallExpr
// CHECK:       DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK:       ImplicitCastExpr {{.*}} 'void *' <BitCast>
// CHECK:         UnaryOperator {{.*}} 'U *' prefix '&'
// CHECK:       ImplicitCastExpr {{.*}} 'const void *' <BitCast>
// CHECK:         UnaryOperator {{.*}} 'U *' prefix '&'
// CHECK:     ReturnStmt
