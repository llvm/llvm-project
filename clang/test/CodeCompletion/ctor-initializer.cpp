struct Base1 {
  Base1() : {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:2:12 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:2:12 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: Pattern : member1(<#args#>)
  // CHECK-CC1: COMPLETION: Pattern : member2(<#args#>

  Base1(int) : member1(123), {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:8:30 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:8:30 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2-NOT: COMPLETION: Pattern : member1(<#args#>)
  // CHECK-CC2: COMPLETION: Pattern : member2(<#args#>

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
// CHECK-CC3: COMPLETION: Pattern : Base1(<#args#>)
// CHECK-CC3: COMPLETION: Pattern : deriv1(<#args#>)

Derived::Derived(int) try : {
} catch (...) {
}
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:31:29 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:31:29 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: COMPLETION: Pattern : Base1(<#args#>)
// CHECK-CC4: COMPLETION: Pattern : deriv1(<#args#>)

Derived::Derived(float) try : Base1(),
{
} catch (...) {
}
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:39:39 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:39:39 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5-NOT: COMPLETION: Pattern : Base1(<#args#>)
// CHECK-CC5: COMPLETION: Pattern : deriv1(<#args#>)

struct A {
  A() : , member2() {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:49:9 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:49:9 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  // CHECK-CC6: COMPLETION: Pattern : member1(<#args#>
  int member1, member2;
};

struct B {
  B() : member2() {}
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:57:9 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++14 -code-completion-at=%s:57:9 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
  // CHECK-CC7: COMPLETION: Pattern : member1(<#args#>
  // Check in the middle and at the end of identifier too.
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:57:13 %s -o - | FileCheck -check-prefix=CHECK-CC8 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++98 -code-completion-at=%s:57:16 %s -o - | FileCheck -check-prefix=CHECK-CC8 %s
  // CHECK-CC8: COMPLETION: Pattern : member2(<#args#>
  int member1, member2;
};
