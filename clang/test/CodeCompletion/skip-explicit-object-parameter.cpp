struct A {
  void foo(this A self, int arg);
};

int main() {
  A a {};
  a.
}
// RUN: %clang_cc1 -cc1 -fsyntax-only -code-completion-at=%s:%(line-2):5 -std=c++23 %s | FileCheck %s
// CHECK: COMPLETION: A : A::
// CHECK-NEXT: COMPLETION: foo : [#void#]foo(<#int arg#>)
// CHECK-NEXT: COMPLETION: operator= : [#A &#]operator=(<#const A &#>)
// CHECK-NEXT: COMPLETION: operator= : [#A &#]operator=(<#A &&#>)
// CHECK-NEXT: COMPLETION: ~A : [#void#]~A()
