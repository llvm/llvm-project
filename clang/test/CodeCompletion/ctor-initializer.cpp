struct Base1 {
  Base1() : {}
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:2:12 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: Pattern : member1(<#args#>)
  // CHECK-CC1: COMPLETION: Pattern : member2(<#args#>

  Base1(int) : member1(123), {}
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:7:30 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
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
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:23:22 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: COMPLETION: Pattern : Base1(<#args#>)
// CHECK-CC3: COMPLETION: Pattern : deriv1(<#args#>)

Derived::Derived(int) try : {
} catch (...) {
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:28:29 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: COMPLETION: Pattern : Base1(<#args#>)
// CHECK-CC4: COMPLETION: Pattern : deriv1(<#args#>)

Derived::Derived(float) try : Base1(),
{
} catch (...) {
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:35:39 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5-NOT: COMPLETION: Pattern : Base1(<#args#>)
// CHECK-CC5: COMPLETION: Pattern : deriv1(<#args#>)
