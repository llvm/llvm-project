// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++2b -ast-dump %s | FileCheck -strict-whitespace %s

namespace GH116928 {
struct S {
  int f(this S&);
};

int main() {
  S s;
  int x = s.f();
  // CHECK: CallExpr 0x{{[^ ]*}} <col:11, col:15> 'int
  // CHECK-NEXT: |-ImplicitCastExpr 0x{{[^ ]*}} <col:13> 'int (*)(S &)' <FunctionToPointerDecay>
  // CHECK-NEXT: | `-DeclRefExpr 0x{{[^ ]*}} <col:13> 'int (S &)' lvalue CXXMethod 0x{{[^ ]*}} 'f' 'int (S &)'
}
}
