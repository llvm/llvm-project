// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++2b -ast-dump %s | FileCheck -strict-whitespace %s

namespace GH116928 {
struct S {
  int f(this S&);
};

void main() {
  S s;
  int x = s.f();
  // CHECK: CallExpr 0x{{[^ ]*}} <col:11, col:15> 'int
  // CHECK-NEXT: |-ImplicitCastExpr 0x{{[^ ]*}} <col:13> 'int (*)(S &)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr 0x{{[^ ]*}} <col:13> 'int (S &)' lvalue CXXMethod 0x{{[^ ]*}} 'f' 'int (S &)'
}
}

namespace GH1269720 {
template <typename T>
struct S {
  void f(this S&);
  void g(S s) {
    s.f();
  }
  // CHECK: CallExpr 0x{{[^ ]*}} <line:22:5, col:9> '<dependent type>'
  // CHECK-NEXT: `-MemberExpr 0x{{[^ ]*}} <col:5, col:7> '<bound member function type>' .f
  // CHECK-NEXT:   `-DeclRefExpr 0x{{[^ ]*}} <col:5> 'S<T>' lvalue ParmVar 0x{{[^ ]*}} 's' 'S<T>'
};
}
