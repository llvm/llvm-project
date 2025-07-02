struct A {
  void foo(this A self, int arg);
};

int main() {
  A a {};
  a.
}
// RUN: %clang_cc1 -cc1 -fsyntax-only -code-completion-at=%s:%(line-2):5 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: COMPLETION: A : A::
// CHECK-NEXT-CC1: COMPLETION: foo : [#void#]foo(<#int arg#>)
// CHECK-NEXT-CC1: COMPLETION: operator= : [#A &#]operator=(<#const A &#>)
// CHECK-NEXT-CC1: COMPLETION: operator= : [#A &#]operator=(<#A &&#>)
// CHECK-NEXT-CC1: COMPLETION: ~A : [#void#]~A()

struct B {
  template <typename T>
  void foo(this T&& self, int arg);
};

int main2() {
  B b {};
  b.foo();
}
// RUN: %clang_cc1 -cc1 -fsyntax-only -code-completion-at=%s:%(line-2):9 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OVERLOAD: [#void#]foo(int arg)
