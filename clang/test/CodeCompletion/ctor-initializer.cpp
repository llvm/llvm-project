struct Base1 {
  Base1() : {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:2:12 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:2:12 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: Pattern : member1(<#int#>)
  // CHECK-CC1: COMPLETION: Pattern : member2(<#float#>)

  Base1(int) : member1(123), {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:8:30 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:8:30 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2-NOT: COMPLETION: Pattern : member1(<#int#>)
  // CHECK-CC2: COMPLETION: Pattern : member2(<#float#>)

  int member1;
  float member2;
};

struct Derived : public Base1 {
  Derived();
  Derived(int);
  Derived(float);
  int deriv1;
};

Derived::Derived() : {}
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:25:22 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:25:22 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: COMPLETION: Pattern : Base1()
// CHECK-CC3: COMPLETION: Pattern : Base1(<#int#>)
// CHECK-CC3: COMPLETION: Pattern : deriv1(<#int#>)

Derived::Derived(int) try : {
} catch (...) {
}
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:32:29 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:32:29 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s

Derived::Derived(float) try : Base1(),
{
} catch (...) {
}
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:38:39 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:38:39 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5-NOT: COMPLETION: Pattern : Base1
// CHECK-CC5: COMPLETION: Pattern : deriv1(<#int#>)

struct A {
  A() : , member2() {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:48:9 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:48:9 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  // CHECK-CC6: COMPLETION: Pattern : member1(<#int#>)
  int member1, member2;
};

struct B {
  B() : member2() {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:56:9 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:56:9 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
  // CHECK-CC7: COMPLETION: Pattern : member1(<#int#>)
  // Check in the middle and at the end of identifier too.
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:56:13 %s -o - | FileCheck -check-prefix=CHECK-CC8 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:56:16 %s -o - | FileCheck -check-prefix=CHECK-CC8 %s
  // CHECK-CC8: COMPLETION: Pattern : member2(<#int#>)
  int member1, member2;
};

struct Base2 {
  Base2(int);
};

struct Composition1 {
  Composition1() : b2_elem(2) {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:72:28 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:72:28 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
  // CHECK-CC9: OVERLOAD: Base2(<#int#>)
  // CHECK-CC9: OVERLOAD: Base2(<#const Base2 &#>)
  // CHECK-CC9-NOT: OVERLOAD: Composition1
  Composition1(Base2);
  Base2 b2_elem;
};

struct Composition2 {
  Composition2() : c1_elem(Base2(1)) {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:83:34 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:83:34 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:83:35 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:83:35 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
  Composition1 c1_elem;
};
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:83:20 %s -o - | FileCheck -check-prefix=CHECK-CC10 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:83:20 %s -o - | FileCheck -check-prefix=CHECK-CC10 %s
// CHECK-CC10: Pattern : c1_elem()
// CHECK-CC10: Pattern : c1_elem(<#Base2#>)

template <class T>
struct Y : T {};

template <class T>
struct X : Y<T> {
  X() : Y<T>() {};
};

// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:100:9 %s -o - | FileCheck -check-prefix=CHECK-CC11 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:100:9 %s -o - | FileCheck -check-prefix=CHECK-CC11 %s
// CHECK-CC11: Pattern : Y<T>(<#Y<T>#>)
