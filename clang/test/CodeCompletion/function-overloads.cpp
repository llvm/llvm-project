int f(int i, int j = 2, int k = 5);
int f(float x, float y...);

class A {
 public:
  A(int, int, int);
};

void test() {
  A a(f(1, 2, 3, 4), 2, 3);
}


namespace NS {
  struct X { };
  struct Y { Y(X); };
  template <class T = int>
  void g(X, Y);
}

void test_adl() {
  NS::X x;
  g(x, x);
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:9 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:10 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:17 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:19 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:20 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:21 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:23:7 %s -o - | \
// RUN:    FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC1: OVERLOAD: [#int#]f(<#float x#>, float y)
// CHECK-CC1: OVERLOAD: [#int#]f(<#int i#>)
// CHECK-CC1-NOT, CHECK-CC2-NOT: OVERLOAD: A(
// CHECK-CC2: OVERLOAD: [#int#]f(float x, float y)
// CHECK-CC2-NOT: OVERLOAD: [#int#]f(int i)
// CHECK-CC3: OVERLOAD: A(<#int#>, int, int)
// CHECK-CC3: OVERLOAD: A(<#const A &#>)
// CHECK-CC3: OVERLOAD: A(<#A &&#>)
// CHECK-CC4: OVERLOAD: A(int, <#int#>, int)
// CHECK-CC5: OVERLOAD: [#void#]g(X, <#Y#>)
