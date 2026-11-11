// Test that the AST records how C-style and functional-style casts were
// interpreted ([expr.cast]p4), and that this survives serialization.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump %s \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump=json %s \
// RUN:   | FileCheck %s --check-prefix=JSON
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -x c++ \
// RUN:   -include-pch %t -ast-dump-all /dev/null | FileCheck %s

struct B {
  virtual ~B();
};
struct D : B {};
using Long = long;

void casts(const int &cr, const int *cip, const B *cb, void *vp,
           const char *cc, int i, float f) {
  (void)(int &)cr;
  // CHECK: CStyleCastExpr {{.*}} 'int' lvalue <NoOp> interpreted_as_const_cast{{$}}

  (void)(int *)cip;
  // CHECK: CStyleCastExpr {{.*}} 'int *' <NoOp> interpreted_as_const_cast{{$}}

  (void)(const int *)cip;
  // CHECK: CStyleCastExpr {{.*}} 'const int *' <NoOp> interpreted_as_const_cast{{$}}

  (void)(D *)cb;
  // CHECK: CStyleCastExpr {{.*}} 'D *' <BaseToDerived (B)> interpreted_as_static_cast_then_const_cast{{$}}

  (void)(int *)vp;
  // CHECK: CStyleCastExpr {{.*}} 'int *' <BitCast>{{$}}

  (void)(int *)cc;
  // CHECK: CStyleCastExpr {{.*}} 'int *' <BitCast> interpreted_as_reinterpret_cast_then_const_cast{{$}}

  (void)(long)&i;
  // CHECK: CStyleCastExpr {{.*}} 'long' <PointerToIntegral> interpreted_as_reinterpret_cast{{$}}

  (void)(int)f;
  // CHECK: CStyleCastExpr {{.*}} 'int' <FloatingToIntegral>{{$}}

  (void)Long(&i);
  // CHECK: CXXFunctionalCastExpr {{.*}} functional cast to Long <PointerToIntegral> interpreted_as_reinterpret_cast{{$}}
}

// JSON: "interpretedAs": "const_cast"
// JSON: "interpretedAs": "static_cast_then_const_cast"
// JSON: "interpretedAs": "reinterpret_cast_then_const_cast"
// JSON: "interpretedAs": "reinterpret_cast"
