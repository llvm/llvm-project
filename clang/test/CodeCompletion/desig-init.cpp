struct Base {
  int t;
};
struct Foo : public Base {
  int x;
  Base b;
  void foo();
};

void foo() {
  Foo F{.x = 2, .b.t = 0};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):10 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC1 %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-2):18 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: b : [#Base#]b
  // CHECK-CC1-NEXT: COMPLETION: x : [#int#]x
  // CHECK-CC1-NOT: foo
  // CHECK-CC1-NOT: t

  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-8):20 %s -o - | FileCheck -check-prefix=CHECK-NESTED %s
  // CHECK-NESTED: COMPLETION: t : [#int#]t

  Base B = {.t = 2};
  auto z = [](Base B) {};
  z({.t = 1});
  z(Base{.t = 2});
  z((Base){.t = 2});
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-5):14 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC2 %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-4):7 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC2 %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-4):11 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC2 %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-4):13 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: COMPLETION: t : [#int#]t
  auto zr = [](const Base &B) {};
  zr({.t = 1});
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):8 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-REF %s
  // CHECK-REF: COMPLETION: t : [#int#]t

  Foo G1{.b = {.t = 0}};
  Foo G2{.b{.t = 0}};
  Foo G3{b: {.t = 0}};
  // RUN: %clang_cc1 -code-completion-at=%s:%(line-3):17 -fsyntax-only -code-completion-patterns %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-NESTED-2 %s
  // RUN: %clang_cc1 -code-completion-at=%s:%(line-3):14 -fsyntax-only -code-completion-patterns %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-NESTED-2 %s
  // RUN: %clang_cc1 -code-completion-at=%s:%(line-3):15 -fsyntax-only -code-completion-patterns %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-NESTED-2 %s
  // CHECK-NESTED-2: COMPLETION: t : [#int#]t
}

// Handle templates
template <typename T>
struct Test { T x; };
template <>
struct Test<int> {
  int x;
  char y;
};
void bar() {
  Test<char> T{.x = 2};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):17 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: COMPLETION: x : [#T#]x
  Test<int> X{.x = 2};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):16 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: COMPLETION: x : [#int#]x
  // CHECK-CC4-NEXT: COMPLETION: y : [#char#]y
}

template <typename T>
void aux() {
  Test<T> X{.x = T(2)};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):14 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC3 %s
}

namespace signature_regression {
  // Verify that an old bug is gone: passing an init-list as a constructor arg
  // would emit overloads as a side-effect.
  struct S{int x;};
  int wrongFunction(S);
  int rightFunction();
  int dummy = wrongFunction({1});
  int x = rightFunction();
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):25 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-SIGNATURE-REGRESSION %s
  // CHECK-SIGNATURE-REGRESSION-NOT: OVERLOAD: [#int#]wrongFunction
  // CHECK-SIGNATURE-REGRESSION:     OVERLOAD: [#int#]rightFunction
  // CHECK-SIGNATURE-REGRESSION-NOT: OVERLOAD: [#int#]wrongFunction
}

struct WithAnon {
  int outer;
  struct { int inner; };
};
auto TestWithAnon = WithAnon { .inner = 2 };
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):33 %s -o - -std=c++2a | FileCheck -check-prefix=CHECK-CC5 %s
  // CHECK-CC5: COMPLETION: inner : [#int#]inner
  // CHECK-CC5: COMPLETION: outer : [#int#]outer
